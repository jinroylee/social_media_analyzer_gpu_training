#!/bin/bash

# Simple RunPod Endpoint Test Script using cURL
# No Python dependencies required!

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 RunPod Endpoint Test Script${NC}"
echo "=================================="

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}📁 Loading variables from .env file...${NC}"
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
else
    echo -e "${YELLOW}⚠️  No .env file found. Using system environment variables...${NC}"
fi

# Configuration - Get from environment variables
ENDPOINT_URL="$RUNPOD_ENDPOINT_URL"
API_KEY="$RUNPOD_API_KEY"
AWS_ACCESS_KEY="$AWS_ACCESS_KEY_ID"
AWS_SECRET_KEY="$AWS_SECRET_ACCESS_KEY"
S3_BUCKET="$S3_BUCKET_NAME"
AWS_REGION="$AWS_REGION"

echo
echo "Configuration loaded:"
echo "ENDPOINT_URL: ${ENDPOINT_URL:-'Not set'}"
echo "API_KEY: ${API_KEY:+${API_KEY:0:10}...}"
echo "AWS_ACCESS_KEY: ${AWS_ACCESS_KEY:+${AWS_ACCESS_KEY:0:10}...}"
echo "AWS_SECRET_KEY: ${AWS_SECRET_KEY:+***set***}"
echo "S3_BUCKET: ${S3_BUCKET:-'Not set'}"
echo "AWS_REGION: ${AWS_REGION:-'Not set'}"
echo

# Check if required values are set
if [[ -z "$ENDPOINT_URL" || "$ENDPOINT_URL" == *"YOUR_ENDPOINT_ID"* ]]; then
    echo -e "${RED}❌ Please set RUNPOD_ENDPOINT_URL in .env file${NC}"
    echo "Format: https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
    exit 1
fi

if [[ -z "$API_KEY" || "$API_KEY" == "your-runpod-api-key" ]]; then
    echo -e "${RED}❌ Please set RUNPOD_API_KEY in .env file${NC}"
    exit 1
fi

if [[ -z "$AWS_ACCESS_KEY" || "$AWS_ACCESS_KEY" == "your-aws-access-key" ]]; then
    echo -e "${RED}❌ Please set AWS_ACCESS_KEY_ID in .env file${NC}"
    exit 1
fi

if [[ -z "$AWS_SECRET_KEY" || "$AWS_SECRET_KEY" == "your-aws-secret-key" ]]; then
    echo -e "${RED}❌ Please set AWS_SECRET_ACCESS_KEY in .env file${NC}"
    exit 1
fi

if [[ -z "$S3_BUCKET" ]]; then
    echo -e "${RED}❌ Please set S3_BUCKET_NAME in .env file${NC}"
    exit 1
fi

if [[ -z "$AWS_REGION" ]]; then
    echo -e "${RED}❌ Please set AWS_REGION in .env file${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All required variables are set${NC}"
echo -e "${YELLOW}📡 Testing endpoint: $ENDPOINT_URL${NC}"
echo -e "${YELLOW}🔑 API Key: ${API_KEY:0:10}...${NC}"
echo

# Create test payload
cat > test_payload.json << EOF
{
  "input": {
    "batch_size": 8,
    "epochs": 2,
    "learning_rate": 0.0001,
    "lora_learning_rate": 0.001,
    "lora_rank": 4,
    "use_lora": true,
    "s3_bucket": "$S3_BUCKET",
    "aws_region": "$AWS_REGION",
    "aws_access_key_id": "$AWS_ACCESS_KEY",
    "aws_secret_access_key": "$AWS_SECRET_KEY"
  }
}
EOF

echo -e "${BLUE}📦 Test payload created with minimal parameters:${NC}"
echo "  - batch_size: 8"
echo "  - epochs: 2"
echo "  - lora_rank: 4"
echo

echo -e "${YELLOW}🧪 Sending test request...${NC}"
echo "This may take a few minutes for training to complete..."
echo

# Send request with timeout
START_TIME=$(date +%s)

response=$(curl -s -w "\n%{http_code}" \
  -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_payload.json \
  --max-time 1800 \
  "$ENDPOINT_URL")

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Parse response
http_code=$(echo "$response" | tail -n1)
response_body=$(echo "$response" | head -n -1)

echo -e "${BLUE}⏱️  Request completed in ${DURATION} seconds${NC}"
echo -e "${BLUE}📊 HTTP Status: $http_code${NC}"
echo

if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✅ SUCCESS! RunPod endpoint is working${NC}"
    echo "=================================================="
    
    # Parse JSON response (basic parsing)
    if echo "$response_body" | grep -q '"status":"success"'; then
        echo -e "${GREEN}🎯 Training Status: SUCCESS${NC}"
        
        # Extract key metrics (basic grep parsing)
        mae=$(echo "$response_body" | grep -o '"best_val_mae":[^,}]*' | cut -d: -f2 | tr -d '"')
        model_path=$(echo "$response_body" | grep -o '"s3_model_path":"[^"]*' | cut -d: -f2- | tr -d '"')
        
        echo -e "${GREEN}📈 Best Validation MAE: $mae${NC}"
        echo -e "${GREEN}💾 Model Path: $model_path${NC}"
        echo
        echo -e "${GREEN}🎉 Your RunPod endpoint is ready for Airflow!${NC}"
        
    elif echo "$response_body" | grep -q '"status":"error"'; then
        echo -e "${RED}❌ Training failed${NC}"
        error_msg=$(echo "$response_body" | grep -o '"message":"[^"]*' | cut -d: -f2- | tr -d '"')
        echo -e "${RED}Error: $error_msg${NC}"
        
        # Show full response for debugging
        echo -e "${YELLOW}Full response:${NC}"
        echo "$response_body" | python3 -m json.tool 2>/dev/null || echo "$response_body"
        
    else
        echo -e "${YELLOW}⚠️  Unexpected response format${NC}"
        echo "$response_body"
    fi
    
else
    echo -e "${RED}❌ FAILED! HTTP $http_code${NC}"
    echo -e "${RED}Response: $response_body${NC}"
    
    # Common error troubleshooting
    if [ "$http_code" = "401" ]; then
        echo -e "${YELLOW}💡 This usually means your API key is invalid${NC}"
    elif [ "$http_code" = "404" ]; then
        echo -e "${YELLOW}💡 This usually means your endpoint URL is incorrect${NC}"
    elif [ "$http_code" = "000" ]; then
        echo -e "${YELLOW}💡 This usually means a connection timeout or network issue${NC}"
    fi
fi

# Cleanup
rm -f test_payload.json

echo
echo "=================================================="
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}🎉 Test completed successfully!${NC}"
    echo
    echo "Next steps for Airflow:"
    echo "1. Set these Airflow Variables:"
    echo "   - runpod_endpoint_url = $ENDPOINT_URL"
    echo "   - runpod_api_key = $API_KEY"
    echo "   - aws_access_key_id = $AWS_ACCESS_KEY"
    echo "   - aws_secret_access_key = [your-secret]"
    echo "2. Deploy your Airflow DAG"
    echo "3. Run your training job!"
else
    echo -e "${RED}❌ Test failed. Please check your configuration.${NC}"
    echo
    echo "Troubleshooting:"
    echo "1. Verify your RunPod endpoint URL"
    echo "2. Check your RunPod API key"
    echo "3. Ensure your AWS credentials are correct"
    echo "4. Make sure your S3 bucket exists and has training data"
fi 