import torch
from torch.utils.data import Dataset

class EngagementDataset(Dataset):
    """Dataset class for engagement prediction with raw images and text."""
    
    def __init__(self, data, processor, tokenizer):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        image_inputs = self.processor(images=item['image'], return_tensors="pt")
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