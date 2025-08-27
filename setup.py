#!/usr/bin/env python3
"""
Setup script for Qwen3 Content Moderation Project

This script sets up the project for development and deployment.

Author: ML Project Team
Date: 2024
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qwen3-content-moderation",
    version="1.0.0",
    author="ML Project Team",
    author_email="contact@example.com",
    description="Content moderation system using fine-tuned Qwen3 model",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/qwen3-content-moderation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
        ],
        "deployment": [
            "docker>=6.0.0",
            "kubernetes>=26.0.0",
            "prometheus-client>=0.16.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "qwen3-moderation-train=scripts.train_model:main",
            "qwen3-moderation-evaluate=scripts.evaluate_model:main",
            "qwen3-moderation-deploy=scripts.deploy_model:main",
            "qwen3-moderation-prepare-data=scripts.prepare_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)
