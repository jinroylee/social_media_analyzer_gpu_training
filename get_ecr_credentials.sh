#!/bin/bash

# Get ECR credentials for RunPod configuration
# Run this script to get the current ECR credentials

set -e

# Configuration
AWS_REGION="ap-northeast-2"
ECR_REGISTRY="777022888924.dkr.ecr.ap-northeast-2.amazonaws.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔐 ECR Credentials for RunPod${NC}"
echo "=================================="
echo ""

# Get ECR login password
echo -e "${YELLOW}📋 Getting ECR login token...${NC}"
ECR_TOKEN=$(aws ecr get-login-password --region ${AWS_REGION})

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully retrieved ECR token${NC}"
    echo ""
    echo -e "${BLUE}📋 Copy these credentials to your RunPod Template:${NC}"
    echo ""
    echo -e "${YELLOW}Registry URL:${NC} ${ECR_REGISTRY}"
    echo -e "${YELLOW}Username:${NC} AWS"
    echo -e "${YELLOW}Password:${NC} ${ECR_TOKEN}"
    echo ""
    echo -e "${GREEN}Instructions:${NC}"
    echo "1. Go to RunPod Console → Templates"
    echo "2. Edit your template or create a new one"
    echo "3. In 'Container Registry Auth' section:"
    echo "   - Username: AWS"
    echo "   - Password: [paste the token above]"
    echo "4. Save and redeploy your endpoint"
    echo ""
    echo -e "${RED}⚠️  Note: ECR tokens expire after 12 hours${NC}"
    echo "   You'll need to update the password periodically"
else
    echo -e "${RED}❌ Failed to get ECR token${NC}"
    echo "Make sure AWS CLI is configured: aws configure"
    exit 1
fi 