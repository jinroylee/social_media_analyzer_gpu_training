# Environment Variables Setup Guide

## Using .env File for RunPod Testing

Instead of hardcoding credentials in your scripts, use a `.env` file to store your configuration securely.

## Setup Steps

### 1. Create your .env file

Copy the example and fill in your actual values:

```bash
cp .env.example .env
```

### 2. Edit .env with your credentials

Open `.env` in your editor and update the values:

```bash
# RunPod Configuration
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ACTUAL_ENDPOINT_ID/runsync
RUNPOD_API_KEY=your-actual-runpod-api-key

# AWS Configuration
AWS_ACCESS_KEY_ID=your-actual-aws-access-key
AWS_SECRET_ACCESS_KEY=your-actual-aws-secret-key
S3_BUCKET_NAME=your-actual-s3-bucket
AWS_REGION=your-aws-region

# Optional: For debugging
DEBUG=false
```

### 3. Secure your .env file

**Important**: Never commit your `.env` file to version control!

Add to your `.gitignore`:
```bash
echo ".env" >> .gitignore
```

## Usage

### Shell Script (test_endpoint.sh)

The shell script automatically loads the `.env` file:

```bash
# Run the test
./test_endpoint.sh
```

The script will:
- Look for `.env` file in the current directory
- Load all variables automatically
- Validate that required variables are set
- Show masked values for security

### Python Script (test_runpod_endpoint.py)

For the Python script, you can optionally install `python-dotenv`:

```bash
pip install python-dotenv
```

Then run:
```bash
python test_runpod_endpoint.py
```

If you don't install `python-dotenv`, the script will still work using system environment variables.

## Alternative: System Environment Variables

You can also set variables in your shell session:

```bash
export RUNPOD_ENDPOINT_URL="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
export RUNPOD_API_KEY="your-runpod-api-key"
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export S3_BUCKET_NAME="your-s3-bucket"
export AWS_REGION="ap-northeast-2"
```

## Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RUNPOD_ENDPOINT_URL` | Your RunPod serverless endpoint URL | `https://api.runpod.ai/v2/abc123/runsync` |
| `RUNPOD_API_KEY` | Your RunPod API key | `ABC123...` |
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3 | `xyz789...` |
| `S3_BUCKET_NAME` | S3 bucket name for training data | `socialmediaanalyzer` |
| `AWS_REGION` | AWS region for S3 bucket | `ap-northeast-2` |

## Security Best Practices

1. **Never commit .env files** - Always add to `.gitignore`
2. **Use different .env files** for different environments (dev, staging, prod)
3. **Restrict file permissions**:
   ```bash
   chmod 600 .env
   ```
4. **Use IAM roles** in production instead of access keys when possible
5. **Rotate credentials regularly**

## Troubleshooting

### "Variables not set" error
- Check that your `.env` file exists in the same directory as the script
- Verify there are no spaces around the `=` sign in your `.env` file
- Make sure variable names match exactly (case-sensitive)

### "Permission denied" error
- Check file permissions: `ls -la .env`
- Fix with: `chmod 600 .env`

### Variables not loading
- Ensure no extra spaces or quotes in `.env` file
- Check for special characters that need escaping
- Verify the file is named exactly `.env` (not `.env.txt`)

## Example .env File

```bash
# RunPod Configuration
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/myendpoint123/runsync
RUNPOD_API_KEY=sk-1234567890abcdef

# AWS Configuration  
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
S3_BUCKET_NAME=my-training-bucket
AWS_REGION=us-east-1
```

Now you can run your tests without hardcoding sensitive information! 