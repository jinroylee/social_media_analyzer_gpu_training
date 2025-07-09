# RunPod ECR Access Setup Guide

## Overview
RunPod needs AWS credentials to pull your Docker image from ECR (Elastic Container Registry). This guide shows you how to create the proper IAM user and credentials.

## Step 1: Create IAM User for RunPod

### 1.1 Go to AWS IAM Console
- Navigate to AWS Console → IAM → Users
- Click "Create user"

### 1.2 Configure User
- **User name**: `runpod-ecr-access` (or similar)
- **Access type**: Select "Programmatic access"
- **Console access**: Not needed (uncheck)

### 1.3 Set Permissions
Choose one of these options:

#### Option A: Use AWS Managed Policy (Recommended)
Attach the policy: `AmazonEC2ContainerRegistryReadOnly`

#### Option B: Create Custom Policy (More Secure)
Create a custom policy with minimal permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        }
    ]
}
```

### 1.4 Get Credentials
After creating the user, you'll get:
- **Access Key ID**: `AKIA...`
- **Secret Access Key**: `xyz123...`

⚠️ **Important**: Save these credentials securely - you won't see the secret key again!

## Step 2: Configure RunPod Template

When creating your RunPod serverless endpoint:

### 2.1 Container Configuration
- **Container Image**: `your-account-id.dkr.ecr.your-region.amazonaws.com/runpod-gpu-training:latest`
- **Container Registry Credentials**: Select "AWS ECR"

### 2.2 AWS Credentials
Enter the credentials from Step 1:
- **AWS Access Key ID**: `AKIA...` (from IAM user)
- **AWS Secret Access Key**: `xyz123...` (from IAM user)
- **AWS Region**: Your ECR region (e.g., `ap-northeast-2`)

## Step 3: Verify Setup

### 3.1 Test Image Pull
RunPod will automatically test pulling your image when you save the template.

### 3.2 Check Logs
If there are issues, check the RunPod logs for ECR authentication errors.

## Security Best Practices

### 1. Principle of Least Privilege
- Use the minimal permissions needed (ECR read-only)
- Don't use your personal AWS credentials
- Create a dedicated IAM user for RunPod

### 2. Credential Management
- Rotate credentials regularly
- Monitor usage in AWS CloudTrail
- Use different credentials for different environments

### 3. ECR Repository Permissions
You can also restrict access to specific ECR repositories:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "ecr:GetAuthorizationToken",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "arn:aws:ecr:your-region:your-account-id:repository/runpod-gpu-training"
        }
    ]
}
```

## Troubleshooting

### Common Issues

#### "Unable to pull image"
- Verify AWS credentials are correct
- Check ECR repository exists and has the right tag
- Ensure IAM user has ECR permissions

#### "Authentication failed"
- Verify Access Key ID and Secret Access Key
- Check if credentials are expired
- Ensure region matches your ECR repository

#### "Repository not found"
- Verify ECR repository name and region
- Check image tag exists
- Ensure repository is in the same region as specified

### Testing ECR Access
You can test ECR access locally:

```bash
# Configure AWS CLI with RunPod credentials
aws configure --profile runpod-test
# Enter the Access Key ID and Secret Access Key

# Test ECR login
aws ecr get-login-password --region your-region --profile runpod-test | docker login --username AWS --password-stdin your-account-id.dkr.ecr.your-region.amazonaws.com

# Test image pull
docker pull your-account-id.dkr.ecr.your-region.amazonaws.com/runpod-gpu-training:latest
```

## Alternative: ECR Public Repository

If you want to avoid AWS credentials entirely, you can:

1. Create an ECR Public repository
2. Push your image to ECR Public
3. Use the public URL in RunPod (no credentials needed)

```bash
# Create ECR Public repository
aws ecr-public create-repository --repository-name runpod-gpu-training --region us-east-1

# Push to ECR Public
docker tag runpod-gpu-training:latest public.ecr.aws/your-alias/runpod-gpu-training:latest
docker push public.ecr.aws/your-alias/runpod-gpu-training:latest
```

## Summary

For RunPod ECR access, you need:
1. **IAM User** with ECR read permissions
2. **Access Key ID** and **Secret Access Key** from that user
3. **AWS Region** where your ECR repository is located

These credentials allow RunPod to authenticate with AWS and pull your Docker image from ECR. 