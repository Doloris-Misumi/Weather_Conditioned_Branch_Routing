import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        print(f"* Loading Text Encoder: {model_name} ...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Project text embedding to target dimension if needed
        # self.projection = nn.Linear(512, target_dim) # CLIP base output is 512

    def forward(self, text_list):
        """
        Args:
            text_list: list of strings, e.g. ["A rainy driving scene", "A sunny driving scene"]
        Returns:
            text_features: (B, 512)
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.text_model.device)
        outputs = self.text_model(**inputs)
        
        # Use the pooled output (CLS token representation)
        text_features = outputs.pooler_output
        
        return text_features

if __name__ == "__main__":
    # Simple test
    encoder = TextEncoder()
    text = ["A heavysnow driving scene at night time", "A normal driving scene at day time"]
    features = encoder(text)
    print("Text features shape:", features.shape)
