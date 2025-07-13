import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import boto3

class EngagementDataset(Dataset):
    """Dataset class for engagement prediction with lazy image loading from S3."""
    
    def __init__(self, data, processor, tokenizer):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        
        # Create S3 client for lazy loading
        if len(data) > 0 and 'aws_config' in data[0]:
            aws_config = data[0]['aws_config']
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_config['aws_access_key_id'],
                aws_secret_access_key=aws_config['aws_secret_access_key'],
                region_name=aws_config['aws_region']
            )
        else:
            self.s3_client = None
    
    def __len__(self):
        return len(self.data)
    
    def load_image_from_s3(self, bucket, key):
        """Load image from S3 on demand"""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response['Body'].read()
            image = Image.open(BytesIO(image_data))
            return image
        except Exception as e:
            # Return a dummy image if loading fails
            return Image.new('RGB', (224, 224), color='black')
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image lazily from S3
        if 'image' in item:
            # Already loaded (old format)
            image = item['image']
        else:
            # Load from S3 (new lazy format)
            image = self.load_image_from_s3(item['s3_bucket'], item['thumbnail_path'])
        
        # Process image
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        # Process text
        text_inputs = self.tokenizer(
            item['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        # Sentiment
        sentiment = torch.tensor([item['sentiment']], dtype=torch.float32)
        
        # Label
        label = torch.tensor(item['label'], dtype=torch.float32)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment': sentiment,
            'label': label
        }