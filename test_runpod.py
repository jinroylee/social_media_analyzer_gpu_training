#!/usr/bin/env python3
"""
Test script for RunPod serverless training function.
This script can be used to test the deployment locally or via API.
"""

import requests
import json
import os
from datetime import datetime
import argparse

def test_local_handler():
    """Test the handler function locally (requires all dependencies)."""
    try:
        # Import the handler
        from runpod_handler import handler
        
        # Create test event
        test_event = {
            "input": {
                "batch_size": 8,  # Small batch for testing
                "epochs": 2,      # Few epochs for testing
                "learning_rate": 1e-4,
                "lora_learning_rate": 1e-3,
                "lora_rank": 4,   # Smaller rank for testing
                "use_lora": True,
                "s3_bucket": os.environ.get('S3_BUCKET', 'your-test-bucket'),
                "aws_region": os.environ.get('AWS_REGION', 'us-east-1'),
                "aws_access_key_id": os.environ.get('AWS_ACCESS_KEY_ID'),
                "aws_secret_access_key": os.environ.get('AWS_SECRET_ACCESS_KEY')
            }
        }
        
        print("Testing local handler...")
        print(f"Test event: {json.dumps(test_event, indent=2)}")
        
        # Run the handler
        result = handler(test_event)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"Local test failed: {e}")
        return None

def test_remote_endpoint(endpoint_url, api_key):
    """Test the RunPod serverless endpoint remotely."""
    try:
        # Request payload
        payload = {
            "input": {
                "batch_size": 16,
                "epochs": 5,
                "learning_rate": 1e-4,
                "lora_learning_rate": 1e-3,
                "lora_rank": 8,
                "use_lora": True,
                "s3_bucket": os.environ.get('S3_BUCKET', 'your-s3-bucket'),
                "aws_region": os.environ.get('AWS_REGION', 'us-east-1'),
                "aws_access_key_id": os.environ.get('AWS_ACCESS_KEY_ID'),
                "aws_secret_access_key": os.environ.get('AWS_SECRET_ACCESS_KEY')
            }
        }
        
        # Headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"Testing remote endpoint: {endpoint_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Make request
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=3600)  # 1 hour timeout
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Remote test failed: {e}")
        return None

def test_minimal_payload(endpoint_url, api_key):
    """Test with minimal payload (only required fields)."""
    try:
        # Minimal payload
        payload = {
            "input": {
                "aws_access_key_id": os.environ.get('AWS_ACCESS_KEY_ID'),
                "aws_secret_access_key": os.environ.get('AWS_SECRET_ACCESS_KEY')
            }
        }
        
        # Headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"Testing minimal payload: {endpoint_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Make request
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=3600)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Minimal test failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test RunPod serverless training function')
    parser.add_argument('--local', action='store_true', help='Test locally')
    parser.add_argument('--remote', help='Test remote endpoint (provide endpoint URL)')
    parser.add_argument('--api-key', help='RunPod API key for remote testing')
    parser.add_argument('--minimal', action='store_true', help='Test with minimal payload')
    
    args = parser.parse_args()
    
    # Check environment variables
    required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Missing environment variables: {missing_vars}")
        print("Please set these environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_value")
        return
    
    if args.local:
        print("=== Testing Local Handler ===")
        result = test_local_handler()
        if result:
            print("✅ Local test passed!")
        else:
            print("❌ Local test failed!")
    
    elif args.remote:
        if not args.api_key:
            print("Error: --api-key is required for remote testing")
            return
        
        print("=== Testing Remote Endpoint ===")
        
        if args.minimal:
            result = test_minimal_payload(args.remote, args.api_key)
        else:
            result = test_remote_endpoint(args.remote, args.api_key)
        
        if result:
            if result.get('status') == 'success':
                print("✅ Remote test passed!")
                training_result = result.get('training_result', {})
                print(f"Best MAE: {training_result.get('best_val_mae', 'N/A')}")
                print(f"Model path: {training_result.get('s3_model_path', 'N/A')}")
            else:
                print("❌ Remote test failed!")
                print(f"Error: {result.get('message', 'Unknown error')}")
        else:
            print("❌ Remote test failed!")
    
    else:
        print("Please specify either --local or --remote <endpoint_url>")
        print("Examples:")
        print("  python test_runpod.py --local")
        print("  python test_runpod.py --remote https://api.runpod.ai/v2/your-endpoint-id/runsync --api-key your-api-key")
        print("  python test_runpod.py --remote https://api.runpod.ai/v2/your-endpoint-id/runsync --api-key your-api-key --minimal")

if __name__ == "__main__":
    main() 