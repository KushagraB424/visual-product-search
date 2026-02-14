import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

# Import the model we just created
from app.ai.feature_extract import FashionEncoder

# --- 1. Custom Dataset that yields Triplets ---
class TripletFashionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects folder structure:
        root_dir/
           class_1/ (e.g. "red_nike_shirt")
               img1.jpg (user photo)
               img2.jpg (shop photo)
           class_2/
               ...
        """
        self.root_dir = root_dir
        self.transform = transform
        # Get all class names (folders)
        self.classes = os.listdir(root_dir)
        # Create a list of all images: [(img_path, class_index), ...]
        self.images = [] 
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    self.images.append((os.path.join(class_path, img_name), i))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Select Anchor
        anchor_path, anchor_label = self.images[idx]
        
        # 2. Select Positive (Same Class, Different Image)
        # In a real dataset, you'd filter more carefully. 
        # Here we just pick another random image from the same folder.
        positive_label = anchor_label
        while True:
            # Simple hack for demo: pick random image, check if label matches
            # Ideally, maintain a dict {label: [list_of_images]}
            rand_idx = random.randint(0, len(self.images)-1)
            path, label = self.images[rand_idx]
            if label == positive_label and path != anchor_path:
                positive_path = path
                break

        # 3. Select Negative (Different Class)
        while True:
            rand_idx = random.randint(0, len(self.images)-1)
            path, label = self.images[rand_idx]
            if label != anchor_label:
                negative_path = path
                break

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

# --- 2. Triplet Loss Function ---
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Euclidean distance
        dist_pos = (anchor - positive).pow(2).sum(1)
        dist_neg = (anchor - negative).pow(2).sum(1)
        # ReLU(d_pos - d_neg + margin)
        loss = torch.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()

# --- 3. Training Loop ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.0001
    EPOCHS = 10

    # Data Transforms (Augmentation is key for fine-tuning!)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Data (Point this to your DeepFashion folder)
    dataset = TripletFashionDataset(root_dir="./data/deepfashion_subset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FashionEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = TripletLoss(margin=1.0)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for anchors, positives, negatives in dataloader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            emb_a = model(anchors)
            emb_p = model(positives)
            emb_n = model(negatives)

            loss = criterion(emb_a, emb_p, emb_n)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fashion_model.pth")
    print("Model saved as fashion_model.pth")

if __name__ == "__main__":
    train()