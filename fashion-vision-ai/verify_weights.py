import torch, timm

state = torch.load("models/weights/classifier.pth", map_location="cpu")
print(f"Keys in state dict: {len(state)}")

# Check if it's a checkpoint dict or plain state_dict
if "model_state_dict" in state:
    print("Detected checkpoint format, extracting model_state_dict...")
    state = state["model_state_dict"]

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=15)
result = model.load_state_dict(state, strict=False)
print(f"Missing keys:    {result.missing_keys}")
print(f"Unexpected keys: {result.unexpected_keys}")

if "classifier.weight" in state:
    print(f"Classifier shape: {state['classifier.weight'].shape}")

print("\nWeights loaded successfully!")
