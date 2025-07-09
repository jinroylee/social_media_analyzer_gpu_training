#!/bin/bash

# RunPod Serverless Deployment Script
# This script helps build and deploy the social media training model to RunPod

set -e

# Configuration - Update these values
DOCKER_REGISTRY="777022888924.dkr.ecr.ap-northeast-2.amazonaws.com"  # e.g., "your-dockerhub-username" or "your-ecr-registry"
IMAGE_NAME="runpod-gpu-training"
TAG="latest"
LOCAL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 RunPod Serverless Deployment Script${NC}"
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "Dockerfile.runpod" ]; then
    echo -e "${RED}❌ Dockerfile.runpod not found. Please run this script from the project root.${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Building Docker image...${NC}"
echo "Local image name: ${LOCAL_IMAGE_NAME}"

# Build the Docker image with local tag first
if docker build -f Dockerfile.runpod -t "${LOCAL_IMAGE_NAME}" .; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Check image size
IMAGE_SIZE=$(docker images "${LOCAL_IMAGE_NAME}" --format "table {{.Size}}" | tail -n 1)
echo -e "${GREEN}📊 Image size: ${IMAGE_SIZE}${NC}"

# Test the image locally (using local tag)
echo -e "${YELLOW}🧪 Testing image locally...${NC}"
echo "Testing basic imports..."

if docker run --rm "${LOCAL_IMAGE_NAME}" python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
import transformers
print(f'Transformers version: {transformers.__version__}')
import runpod
print('RunPod imported successfully')
import modelfactory
print('✅ modelfactory imported successfully')
"; then
    echo -e "${GREEN}✅ Local test passed${NC}"
else
    echo -e "${RED}❌ Local test failed${NC}"
    exit 1
fi

# Authenticate with ECR
echo -e "${YELLOW}🔐 Authenticating with ECR...${NC}"
AWS_REGION="ap-northeast-2"  # Extract region from registry URL
if aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}; then
    echo -e "${GREEN}✅ ECR authentication successful${NC}"
else
    echo -e "${RED}❌ ECR authentication failed. Make sure AWS CLI is configured and you have ECR permissions.${NC}"
    echo "Run: aws configure"
    exit 1
fi

# Tag the image for ECR
echo -e "${YELLOW}🏷️  Tagging image for ECR...${NC}"
if docker tag "${LOCAL_IMAGE_NAME}" "${FULL_IMAGE_NAME}"; then
    echo -e "${GREEN}✅ Image tagged for ECR: ${FULL_IMAGE_NAME}${NC}"
else
    echo -e "${RED}❌ Failed to tag image${NC}"
    exit 1
fi

# Push to registry
echo -e "${YELLOW}📤 Pushing to ECR...${NC}"
if docker push "${FULL_IMAGE_NAME}"; then
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
else
    echo -e "${RED}❌ Push failed. Check your ECR permissions and repository exists.${NC}"
    echo "Create ECR repository with: aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${AWS_REGION}"
    exit 1
fi

echo -e "${GREEN}🎉 Deployment preparation complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Go to RunPod console: https://www.runpod.io/console"
echo "2. Create a new template with image: ${FULL_IMAGE_NAME}"
echo "3. Configure GPU settings (recommend RTX 4090 or A100)"
echo "4. Create a serverless endpoint using the template"
echo "5. Test the endpoint using the provided examples"
echo ""
echo "Image details:"
echo "  Registry: ${DOCKER_REGISTRY}"
echo "  Image: ${IMAGE_NAME}"
echo "  Tag: ${TAG}"
echo "  Full name: ${FULL_IMAGE_NAME}"
echo "  Size: ${IMAGE_SIZE}"
echo ""
echo "Test your deployment with:"
echo "  python test_runpod.py --remote <your-endpoint-url> --api-key <your-api-key>" 