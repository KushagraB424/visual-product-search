import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class FashionEncoder(nn.Module):
    def __init__(self, embedding_dim=2048, pretrained=True):
        super(FashionEncoder, self).__init__()
        # Load ResNet50 trained on ImageNet
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Remove the final classification layer (fc)
        # ResNet50's last layer inputs 2048 features
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Pass through raw features
        
        # Add a custom embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        
        # L2 Normalize embeddings (Crucial for Cosine Similarity!)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

# --- Helper for Inference (What the API will use) ---
# We define standard transforms here so training and inference match
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path=None, device="cpu"):
    model = FashionEncoder()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_features(image_bytes, model, device="cpu"):
    """
    Takes raw image bytes -> Returns 2048-d list
    """
    image = Image.open(image_bytes).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(tensor)
    
    return embedding.cpu().numpy().flatten().tolist()