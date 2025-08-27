#!/usr/bin/env python3
"""
Model Deployment Script for Qwen3 Content Moderation

This script deploys the fine-tuned Qwen3 content moderation model
as a REST API service for production use.

Author: ML Project Team
Date: 2024
"""

import os
import sys
import json
import yaml
import argparse
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import re

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Request/Response Models
class ModerationRequest(BaseModel):
    content: str = Field(..., description="Content to moderate", max_length=10000)
    include_reasoning: bool = Field(default=True, description="Include reasoning in response")
    custom_thresholds: Optional[Dict[str, float]] = Field(default=None, description="Custom safety thresholds")

class ModerationResponse(BaseModel):
    classification: str = Field(..., description="Overall classification (SAFE/UNSAFE)")
    categories: List[str] = Field(..., description="Detected unsafe categories")
    severity: str = Field(..., description="Severity level (safe/low/medium/high/critical)")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for classification")
    processing_time: float = Field(..., description="Processing time in seconds")

class BatchModerationRequest(BaseModel):
    contents: List[str] = Field(..., description="List of content to moderate", max_items=100)
    include_reasoning: bool = Field(default=False, description="Include reasoning in responses")

class BatchModerationResponse(BaseModel):
    results: List[ModerationResponse] = Field(..., description="Moderation results")
    total_processing_time: float = Field(..., description="Total processing time in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    uptime: float = Field(..., description="Service uptime in seconds")

class ContentModerationService:
    """Content moderation service using Qwen3 model."""
    
    def __init__(self, model_path: str, config_path: str = "config/deployment_config.yaml"):
        """Initialize the service."""
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.model_config = self._load_model_config()
        
        # Service state
        self.model = None
        self.tokenizer = None
        self.start_time = time.time()
        self.request_count = 0
        self.categories = list(self.model_config['capabilities']['content_categories'].keys())
        
        # Safety thresholds
        self.default_thresholds = self.model_config['safety_rules']['category_thresholds']
        
        logger.info(f"Initializing content moderation service with model: {model_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load deployment configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading deployment config: {e}")
            raise
    
    def _load_model_config(self) -> Dict:
        """Load model configuration."""
        try:
            with open("config/model_config.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            raise
    
    async def load_model(self):
        """Load the model and tokenizer."""
        logger.info("Loading model and tokenizer...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model
            device_config = self.config['infrastructure']['compute']
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Check if it's a PEFT model
            if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
                logger.info("Loading PEFT adapter...")
                base_model_path = self._get_base_model_path()
                if base_model_path:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_base_model_path(self) -> Optional[str]:
        """Get base model path from adapter config."""
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                return adapter_config.get("base_model_name_or_path")
        return None
    
    def create_prompt(self, content: str) -> str:
        """Create moderation prompt."""
        template = self.model_config['input_output']['input_template']
        return template.format(content=content)
    
    async def generate_response(self, prompt: str) -> str:
        """Generate model response."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            assistant_start = full_response.find("<|im_start|>assistant\n")
            if assistant_start != -1:
                response = full_response[assistant_start + len("<|im_start|>assistant\n"):]
                response = response.replace("<|im_end|>", "").strip()
                return response
            else:
                return full_response[len(prompt):].strip()
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def parse_response(self, response: str) -> Dict:
        """Parse model response."""
        result = {
            'classification': 'UNKNOWN',
            'categories': [],
            'severity': 'unknown',
            'confidence': 0.0,
            'reasoning': ''
        }
        
        try:
            # Extract classification
            class_match = re.search(r'Classification:\s*(SAFE|UNSAFE)', response, re.IGNORECASE)
            if class_match:
                result['classification'] = class_match.group(1).upper()
            
            # Extract categories
            cat_match = re.search(r'Categories:\s*([^\n]+)', response, re.IGNORECASE)
            if cat_match:
                categories_str = cat_match.group(1)
                categories = [cat.strip() for cat in categories_str.split(',')]
                result['categories'] = [cat for cat in categories if cat in self.categories]
            
            # Extract severity
            sev_match = re.search(r'Severity:\s*(\w+)', response, re.IGNORECASE)
            if sev_match:
                result['severity'] = sev_match.group(1).lower()
            
            # Extract confidence
            conf_match = re.search(r'Confidence:\s*([\d.]+)', response, re.IGNORECASE)
            if conf_match:
                result['confidence'] = float(conf_match.group(1))
            
            # Extract reasoning
            reason_match = re.search(r'Reasoning:\s*([^\n]+)', response, re.IGNORECASE)
            if reason_match:
                result['reasoning'] = reason_match.group(1).strip()
        
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
        
        return result
    
    async def moderate_content(self, content: str, include_reasoning: bool = True,
                             custom_thresholds: Optional[Dict[str, float]] = None) -> ModerationResponse:
        """Moderate a single piece of content."""
        start_time = time.time()
        
        try:
            # Validate input
            if not content or not content.strip():
                raise HTTPException(status_code=400, detail="Content cannot be empty")
            
            if len(content) > 10000:
                raise HTTPException(status_code=400, detail="Content too long (max 10,000 characters)")
            
            # Generate moderation result
            prompt = self.create_prompt(content)
            response = await self.generate_response(prompt)
            parsed = self.parse_response(response)
            
            # Apply custom thresholds if provided
            if custom_thresholds:
                # Adjust classification based on custom thresholds
                # This is a simplified implementation
                pass
            
            processing_time = time.time() - start_time
            self.request_count += 1
            
            return ModerationResponse(
                classification=parsed['classification'],
                categories=parsed['categories'],
                severity=parsed['severity'],
                confidence=parsed['confidence'],
                reasoning=parsed['reasoning'] if include_reasoning else None,
                processing_time=processing_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error moderating content: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def moderate_batch(self, contents: List[str], include_reasoning: bool = False) -> BatchModerationResponse:
        """Moderate multiple pieces of content."""
        start_time = time.time()
        
        if len(contents) > 100:
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
        
        results = []
        for content in contents:
            try:
                result = await self.moderate_content(content, include_reasoning)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch moderation: {e}")
                # Add error result
                results.append(ModerationResponse(
                    classification="ERROR",
                    categories=[],
                    severity="unknown",
                    confidence=0.0,
                    reasoning="Processing error",
                    processing_time=0.0
                ))
        
        total_time = time.time() - start_time
        
        return BatchModerationResponse(
            results=results,
            total_processing_time=total_time
        )
    
    def get_health_status(self) -> HealthResponse:
        """Get service health status."""
        # Check memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Check GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    "cached": torch.cuda.memory_reserved(i) / 1024**3  # GB
                }
        
        return HealthResponse(
            status="healthy" if self.model is not None else "loading",
            model_loaded=self.model is not None,
            gpu_available=torch.cuda.is_available(),
            memory_usage={
                "ram_gb": memory_info.rss / 1024**3,
                "gpu_memory": gpu_memory
            },
            uptime=time.time() - self.start_time
        )

# Global service instance
service = None

# FastAPI app
app = FastAPI(
    title="Qwen3 Content Moderation API",
    description="Content moderation service using fine-tuned Qwen3 model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    global service
    model_path = os.getenv("MODEL_PATH", "models/final/qwen3-content-moderation")
    service = ContentModerationService(model_path)
    await service.load_model()
    logger.info("Content moderation service started successfully")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return service.get_health_status()

@app.post("/api/v1/moderate", response_model=ModerationResponse)
async def moderate_content(request: ModerationRequest):
    """Moderate a single piece of content."""
    return await service.moderate_content(
        content=request.content,
        include_reasoning=request.include_reasoning,
        custom_thresholds=request.custom_thresholds
    )

@app.post("/api/v1/moderate/batch", response_model=BatchModerationResponse)
async def moderate_batch(request: BatchModerationRequest):
    """Moderate multiple pieces of content."""
    return await service.moderate_batch(
        contents=request.contents,
        include_reasoning=request.include_reasoning
    )

@app.get("/api/v1/stats")
async def get_stats():
    """Get service statistics."""
    return {
        "total_requests": service.request_count,
        "uptime_seconds": time.time() - service.start_time,
        "model_path": service.model_path,
        "categories": service.categories
    }

@app.get("/api/v1/categories")
async def get_categories():
    """Get available content categories."""
    return {
        "categories": service.categories,
        "descriptions": {
            cat: service.model_config['capabilities']['content_categories'][cat]['description']
            for cat in service.categories
        }
    }

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Qwen3 Content Moderation Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/final/qwen3-content-moderation",
        help="Path to the trained model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set environment variable for model path
    os.environ["MODEL_PATH"] = args.model_path
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"Starting content moderation service on {args.host}:{args.port}")
    logger.info(f"Model path: {args.model_path}")
    
    # Run the server
    uvicorn.run(
        "deploy_model:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False
    )

if __name__ == "__main__":
    main()
