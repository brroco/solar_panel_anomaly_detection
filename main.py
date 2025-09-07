import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import patchcore
import patchcore.backbones

# 1. Load your images (rectangular, e.g., 256x384)
transform = transforms.Compose([
    transforms.Resize((224,224)),  # Ensure all images are the same size
    transforms.ToTensor()
])

train_dataset = ImageFolder("data/train", transform=transform)
test_dataset = ImageFolder("data/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 2. Configure PatchCore
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = patchcore.PatchCore(device)
backbone = patchcore.backbones.load("resnet50")
layers = ["layer2", "layer3"]  # Use layers appropriate for your backbone

model.load(
    backbone=backbone,
    layers_to_extract_from=layers,
    device=device,
    input_shape=(3, 256, 384),  # Channels, Height, Width
    pretrain_embed_dimension=1024,
    target_embed_dimension=384,
    patchsize=3,
    patchstride=1,
)

# 3. Train PatchCore (build memory bank)
model.fit(train_loader)

# 4. Predict anomalies on test set
scores, masks, labels_gt, masks_gt = model.predict(test_loader)

# 5. (Optional) Visualize or save results
for i, (score, mask) in enumerate(zip(scores, masks)):
    print(f"Image {i}: Anomaly score = {score}")
    # You can save or display mask here