import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model, TaskType

class EngagementHead(nn.Module):
    def __init__(self, input_dim=1537, hidden_dim=1024):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class CLIPEngagementRegressor(nn.Module):
    def __init__(self, use_lora=True, lora_rank=8):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        
        if use_lora:
            # Configure LoRA for CLIP model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=16,  # Scaling factor (typically 2*rank)
                target_modules=[
                    # Vision transformer modules
                    "q_proj", "k_proj", "v_proj", "out_proj",
                    # Text transformer modules  
                    "q_proj", "k_proj", "v_proj", "out_proj",
                    # MLP modules
                    "fc1", "fc2"
                ],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            # Apply LoRA to the CLIP model
            self.clip_model = get_peft_model(self.clip_model, lora_config)
            print(f"LoRA applied to CLIP model with rank {lora_rank}")
            self.clip_model.print_trainable_parameters()
        
        self.head = EngagementHead()

    def forward(self, image, input_ids, attention_mask, sentiment):
        image_embeds = self.clip_model.get_image_features(pixel_values=image)
        text_embeds = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        concat = torch.cat([image_embeds, text_embeds, sentiment], dim=1)
        return self.head(concat)