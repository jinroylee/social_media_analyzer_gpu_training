import os
import math
import re
import pandas as pd
import torch
from transformers import pipeline
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from modelfactory.utils.data_stat import visualize_distribution
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from io import BytesIO

# Load environment variables
load_dotenv()

###########################
w_like = 1
w_comment = 10
w_share = 100
###########################

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Only load sentiment pipeline for preprocessing
sentiment_pipeline = pipeline(
    "text-classification", 
    model="tabularisai/multilingual-sentiment-analysis",
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    batch_size=32,  # Process multiple texts at once
    return_all_scores=False
)

EMOJI_PATTERN = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
URL_PATTERN = re.compile(r"http\S+")

sentiment_map = {"Very Negative":0, "Negative":1, "Neutral":2, "Positive":3, "Very Positive":4}
    
def load_parquet_from_s3(bucket_name, key):
    """Load parquet file from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return pd.read_parquet(BytesIO(response['Body'].read()))
    except ClientError as e:
        print(f"Error loading parquet from S3: {e}")
        raise

def load_image_from_s3(bucket_name, key):
    """Load image from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_data = BytesIO(response['Body'].read())
        return Image.open(image_data).convert("RGB")
    except ClientError as e:
        print(f"Error loading image from S3 {key}: {e}")
        return None
    except Exception as e:
        print(f"Error opening image {key}: {e}")
        return None
    
def clean_text(text):
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_sentiment(comments):
    """
    Compute sentiment score as discrete value from 0-4.
    If no comments exist, return middle value (2).
    """
    if len(comments) == 0:
        return 2  # Middle value for neutral when no comments
    
    # Join all comments into a single string for sentiment analysis
    combined_text = " ".join(comments)
    
    # Get prediction from the model
    result = sentiment_pipeline(combined_text)
    # print("comments: ", combined_text)
    # print("SA result: ", result)
    # Check if 'label' exists in the first element of result
    if 'label' in result[0]:
        label = result[0]['label']
        sentiment_class = sentiment_map[label]
        # print("true label: ", label)
        # print("sentiment class: ", sentiment_class)
    else:
        sentiment_class = 2  # Default to neutral if 'label' is not found
    
    return sentiment_class

def compute_engagement(row):
    followers = max(int(row['follower_count']), 1)
    views = max(int(row['view_count']), 1)
    like_rate = int(row['like_count']) / views
    comment_rate = int(row['comment_count']) / views
    share_rate = int(row['share_count']) / views
    reach_boost = math.log(1 + views)/math.log(1 + followers)

    # print(like_rate, comment_rate, share_rate, reach_boost)
    score = (w_like*like_rate + w_comment*comment_rate + w_share*share_rate) * reach_boost
    #score = views/followers
    # print("score: ", score)
    return np.log1p(score)

def normalize_sentiment(sentiment_values):
    return np.array(sentiment_values) / 4.0

def normalize_engagement_labels(engagement_scores):
    scaler = StandardScaler()
    print("engagement_scores: ", engagement_scores[:10])
    engagement_scores = np.clip(engagement_scores, -5, 5)
    visualize_distribution(engagement_scores, "Raw Engagement Scores Distribution")
    normalized_scores = scaler.fit_transform(np.array(engagement_scores).reshape(-1, 1))
    print("normalized_scores: ", normalized_scores[:10])
    return normalized_scores.flatten()

def prepare_data(df):
    """
    Prepare data for training by saving raw inputs instead of pre-computed embeddings.
    This allows fine-tuning of the CLIP encoders.
    """
    data = []
    sentiment_scores = []
    engagement_scores = []
    skipped_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Extract thumbnail filename from path and construct S3 key
        thumbnail_filename = os.path.basename(row['thumbnail_path'])
        s3_key = f"raw/thumbnails/{thumbnail_filename}"
        
        # Load image from S3
        image = load_image_from_s3(S3_BUCKET_NAME, s3_key)
        if image is None:
            skipped_count += 1
            continue  # Skip this row if image doesn't exist or can't be loaded
        
        # Clean text
        description = clean_text(row['description'])
        
        # Compute sentiment (discrete 0-4)
        sentiment_score = compute_sentiment(row['top_comments'])
        sentiment_scores.append(sentiment_score)
        
        # Compute engagement label
        engagement_score = compute_engagement(row)
        engagement_scores.append(engagement_score)
        
        # Store raw data for training (will normalize after collecting all scores)
        data_point = {
            'video_id': row['video_id'],
            'thumbnail_path': s3_key,  # Store S3 key instead of local path
            'raw_text': row['description'],
            'top_comments': row['top_comments'],
            'view_count': row['view_count'],
            'like_count': row['like_count'],
            'comment_count': row['comment_count'],
            'share_count': row['share_count'],
            'image': image,
            'text': description,
            'sentiment': sentiment_score,
            'label': engagement_score
        }
        data.append(data_point)
    
    print(f"Skipped {skipped_count} rows due to missing or corrupted images")
    
    # Normalize sentiment scores (0-4 -> 0-1 with 2->0.5)
    normalized_sentiments = normalize_sentiment(sentiment_scores)
    
    # MinMax normalize engagement labels
    normalized_engagement = normalize_engagement_labels(engagement_scores)
    
    # Update data with normalized values
    for i, data_point in enumerate(data):
        data_point['sentiment'] = normalized_sentiments[i]
        data_point['label'] = normalized_engagement[i]
    
    return data

def split_and_save_data(data, train_ratio=0.9, random_state=42):
    """
    Split data into training and testing sets and save them to S3.
    
    Args:
        data: List of processed data points
        train_ratio: Ratio for training data (default 0.9 for 9:1 split)
        random_state: Random seed for reproducibility
    """
    print(f"\nSplitting data into {train_ratio:.1%} training and {1-train_ratio:.1%} testing...")
    
    # Split the data randomly
    train_data, test_data = train_test_split(
        data, 
        train_size=train_ratio, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Save training data to S3
    train_key = "processed/train.pkl"
    if save_pkl_to_s3(train_data, S3_BUCKET_NAME, train_key):
        print(f"Training data saved to s3://{S3_BUCKET_NAME}/{train_key}")
    else:
        print("Failed to save training data to S3")
    
    # Save testing data to S3
    test_key = "processed/test.pkl"
    if save_pkl_to_s3(test_data, S3_BUCKET_NAME, test_key):
        print(f"Testing data saved to s3://{S3_BUCKET_NAME}/{test_key}")
    else:
        print("Failed to save testing data to S3")
    
    # Also save the complete dataset for backward compatibility
    complete_key = "processed/complete_data.pkl"
    if save_pkl_to_s3(data, S3_BUCKET_NAME, complete_key):
        print(f"Complete dataset saved to s3://{S3_BUCKET_NAME}/{complete_key}")
    else:
        print("Failed to save complete dataset to S3")
    
    return train_data, test_data

def save_pkl_to_s3(data, bucket_name, key):
    """Save pickle data to S3"""
    try:
        # Serialize data to bytes
        pkl_buffer = BytesIO()
        pickle.dump(data, pkl_buffer)
        pkl_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=pkl_buffer.getvalue()
        )
        print(f"Successfully saved {key} to S3")
        return True
    except Exception as e:
        print(f"Error saving {key} to S3: {e}")
        return False

def main():
    print("Starting data preprocessing...")
    print(f"Loading data from S3 bucket: {S3_BUCKET_NAME}")
    print(f"Using credential: {boto3.client('sts').get_caller_identity()}")
    # Load parquet data from S3
    df = load_parquet_from_s3(S3_BUCKET_NAME, "raw/data/tiktok_data.parquet")

    print("Data loaded successfully from S3")
    data = prepare_data(df)
    
    # Split and save data
    train_data, test_data = split_and_save_data(data, train_ratio=0.9, random_state=42)
    
    print(f"\nProcessed {len(data)} total samples")
    
    # Print some statistics for training data
    train_sentiments = [d['sentiment'] for d in train_data]
    train_labels = [d['label'] for d in train_data]
    
    print(f"\nTraining data statistics:")
    print(f"  Sentiment - Min: {min(train_sentiments):.3f}, Max: {max(train_sentiments):.3f}")
    print(f"  Sentiment - Mean: {np.mean(train_sentiments):.3f}, Std: {np.std(train_sentiments):.3f}")
    print(f"  Engagement - Min: {min(train_labels):.3f}, Max: {max(train_labels):.3f}")
    print(f"  Engagement - Mean: {np.mean(train_labels):.3f}, Std: {np.std(train_labels):.3f}")

if __name__ == "__main__":
    main()
