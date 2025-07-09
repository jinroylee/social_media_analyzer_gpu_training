#!/usr/bin/env python3
"""
Simple RunPod Endpoint Test Script
Test your RunPod serverless endpoint before using it in Airflow
"""

import requests
import json
import time
import os
from datetime import datetime

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("📁 Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")
    print("   Install with: pip install python-dotenv")

def test_runpod_endpoint():
    """Test RunPod endpoint with minimal parameters"""
    
    # Configuration - Load from environment variables
    ENDPOINT_URL = os.getenv('RUNPOD_ENDPOINT_URL')
    API_KEY = os.getenv('RUNPOD_API_KEY')
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'socialmediaanalyzer')
    AWS_REGION = os.getenv('AWS_REGION', 'ap-northeast-2')
    
    # Validate required environment variables
    if not ENDPOINT_URL or 'YOUR_ENDPOINT_ID' in ENDPOINT_URL:
        print("❌ Please set RUNPOD_ENDPOINT_URL in .env file")
        print("   Format: https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync")
        return False
    
    if not API_KEY or API_KEY == 'your-runpod-api-key':
        print("❌ Please set RUNPOD_API_KEY in .env file")
        return False
    
    if not AWS_ACCESS_KEY or AWS_ACCESS_KEY == 'your-aws-access-key':
        print("❌ Please set AWS_ACCESS_KEY_ID in .env file")
        return False
    
    if not AWS_SECRET_KEY or AWS_SECRET_KEY == 'your-aws-secret-key':
        print("❌ Please set AWS_SECRET_ACCESS_KEY in .env file")
        return False
    
    # Test payload with minimal parameters
    test_payload = {
        "input": {
            "batch_size": 8,  # Small batch for testing
            "epochs": 2,      # Few epochs for quick test
            "learning_rate": 1e-4,
            "lora_learning_rate": 1e-3,
            "lora_rank": 4,   # Smaller rank for testing
            "use_lora": True,
            "s3_bucket": S3_BUCKET,
            "aws_region": AWS_REGION,
            "aws_access_key_id": AWS_ACCESS_KEY,
            "aws_secret_access_key": AWS_SECRET_KEY
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("\n🧪 Testing RunPod Endpoint")
    print("=" * 50)
    print(f"Endpoint: {ENDPOINT_URL}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"AWS Region: {AWS_REGION}")
    print(f"Test parameters: batch_size={test_payload['input']['batch_size']}, epochs={test_payload['input']['epochs']}")
    print()
    
    try:
        print("📡 Sending request to RunPod...")
        start_time = time.time()
        
        response = requests.post(
            ENDPOINT_URL,
            json=test_payload,
            headers=headers,
            timeout=1800  # 30 minutes timeout for testing
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Request completed in {duration:.1f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! RunPod endpoint is working")
            print("=" * 50)
            
            # Parse and display results
            if result.get('status') == 'success':
                training_result = result.get('training_result', {})
                
                print(f"🎯 Training Status: {result.get('status')}")
                print(f"📈 Best Validation MAE: {training_result.get('best_val_mae', 'N/A')}")
                print(f"🔄 Total Epochs: {training_result.get('total_epochs', 'N/A')}")
                print(f"💾 Model Path: {training_result.get('s3_model_path', 'N/A')}")
                print(f"⚙️  Config Path: {training_result.get('s3_config_path', 'N/A')}")
                print(f"📤 Model Uploaded: {training_result.get('model_uploaded', 'N/A')}")
                print(f"🕐 Timestamp: {result.get('timestamp', 'N/A')}")
                
                print("\n🎉 Your RunPod endpoint is ready for Airflow!")
                
            elif result.get('status') == 'error':
                print(f"❌ Training failed: {result.get('message', 'Unknown error')}")
                if result.get('traceback'):
                    print(f"🔍 Error details:\n{result.get('traceback')}")
                return False
            else:
                print(f"⚠️  Unexpected response format: {result}")
                return False
                
        else:
            print(f"❌ FAILED! HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (30 minutes)")
        print("This might be normal for longer training jobs")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - check your endpoint URL")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_async_endpoint():
    """Test RunPod async endpoint (for longer jobs)"""
    
    # Configuration - Load from environment variables
    ENDPOINT_URL = os.getenv('RUNPOD_ENDPOINT_URL')
    API_KEY = os.getenv('RUNPOD_API_KEY')
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'socialmediaanalyzer')
    AWS_REGION = os.getenv('AWS_REGION', 'ap-northeast-2')
    
    # Validate required environment variables
    if not ENDPOINT_URL or 'YOUR_ENDPOINT_ID' in ENDPOINT_URL:
        print("❌ Please set RUNPOD_ENDPOINT_URL in .env file")
        return False
    
    if not API_KEY or API_KEY == 'your-runpod-api-key':
        print("❌ Please set RUNPOD_API_KEY in .env file")
        return False
    
    # Convert sync endpoint to async
    ENDPOINT_URL = ENDPOINT_URL.replace('/runsync', '/run')
    
    test_payload = {
        "input": {
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 1e-4,
            "lora_learning_rate": 1e-3,
            "lora_rank": 4,
            "use_lora": True,
            "s3_bucket": S3_BUCKET,
            "aws_region": AWS_REGION,
            "aws_access_key_id": AWS_ACCESS_KEY,
            "aws_secret_access_key": AWS_SECRET_KEY
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing RunPod Async Endpoint")
    print("=" * 50)
    print(f"Endpoint: {ENDPOINT_URL}")
    print()
    
    try:
        # Start async job
        print("📡 Starting async job...")
        response = requests.post(ENDPOINT_URL, json=test_payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data.get('id')
            
            if job_id:
                print(f"✅ Job started successfully!")
                print(f"🆔 Job ID: {job_id}")
                print(f"📊 Status: {job_data.get('status', 'Unknown')}")
                print()
                print("💡 To check job status later, use:")
                print(f"   GET {ENDPOINT_URL.replace('/run', '')}/status/{job_id}")
                print()
                print("🔄 You can now use this endpoint in Airflow with async polling")
                return True
            else:
                print(f"❌ No job ID returned: {job_data}")
                return False
        else:
            print(f"❌ Failed to start job: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def quick_health_check():
    """Quick health check of the endpoint"""
    
    ENDPOINT_URL = os.getenv('RUNPOD_ENDPOINT_URL', 'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync')
    API_KEY = os.getenv('RUNPOD_API_KEY', 'your-runpod-api-key')
    
    # Simple ping test
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    print("🏥 Quick Health Check")
    print("=" * 30)
    
    try:
        # Try to reach the endpoint (this might fail but tells us about connectivity)
        response = requests.get(ENDPOINT_URL.replace('/runsync', '/health'), headers=headers, timeout=10)
        print(f"📡 Endpoint reachable: {response.status_code}")
    except:
        print("📡 Endpoint connectivity: Cannot determine (normal for RunPod)")
    
    print(f"🔗 Endpoint URL: {ENDPOINT_URL}")
    print(f"🔑 API Key: {'✅ Set' if API_KEY != 'your-runpod-api-key' else '❌ Not set'}")
    
    if API_KEY == 'your-runpod-api-key':
        print("\n⚠️  Please update your API key before testing!")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 RunPod Endpoint Test Suite")
    print("=" * 60)
    print()
    
    # Quick health check first
    if not quick_health_check():
        print("\n❌ Please configure your endpoint URL and API key first!")
        exit(1)
    
    print("\nChoose test type:")
    print("1. Sync test (recommended for first test)")
    print("2. Async test (for longer jobs)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        success = test_runpod_endpoint()
    elif choice == "2":
        success = test_async_endpoint()
    elif choice == "3":
        print("\n" + "="*60)
        success1 = test_runpod_endpoint()
        print("\n" + "="*60)
        success2 = test_async_endpoint()
        success = success1 and success2
    else:
        print("Invalid choice")
        exit(1)
    
    print("\n" + "="*60)
    if success:
        print("🎉 All tests passed! Your endpoint is ready for Airflow.")
        print("\nNext steps:")
        print("1. Set these Airflow Variables:")
        print("   - runpod_endpoint_url")
        print("   - runpod_api_key")
        print("   - aws_access_key_id")
        print("   - aws_secret_access_key")
        print("2. Deploy your Airflow DAG")
        print("3. Run your training job!")
    else:
        print("❌ Some tests failed. Please check your configuration.") 