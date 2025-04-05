import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ----- Config -----
IMAGE_DIR = 'composited'
MASK_DIR = 'masks'
EPOCHS = 50
BATCH_SIZE = 4
NUM_CLASSES = 2  # 1 class (fruit) + background
MAX_IMAGES = 2000  # Cap to speed up training
IMG_SIZE = (256, 256)  # Downsample to speed up training

# ----- Custom Dataset -----
class FruitSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        self.images = []
        for file in os.listdir(image_dir):
            if file.endswith('.png'):
                image_path = os.path.join(image_dir, file)
                mask_path = os.path.join(mask_dir, file.replace('comp_', 'mask_'))
                self.images.append((image_path, mask_path))
        self.images = self.images[:MAX_IMAGES]

    def __getitem__(self, idx):
        img_path, mask_path = self.images[idx]
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        mask = Image.open(mask_path).convert("L").resize(IMG_SIZE)

        img = F.to_tensor(img)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        masks = mask == obj_ids[:, None, None]
        boxes = []
        for m in masks:
            pos = np.where(m)
            if pos[0].size == 0 or pos[1].size == 0:
                continue  # skip empty masks
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax <= xmin or ymax <= ymin:
                continue  # skip invalid boxes
            boxes.append([xmin, ymin, xmax, ymax])

        if not boxes:
            # fallback: dummy box to avoid crash
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, *mask.shape), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

# ----- Utils -----
def collate_fn(batch):
    return tuple(zip(*batch))

# ----- Main Training Loop -----
def train(fruit):
    print("[BOOT] TrainMRCNN.py script is running...")
    print("[INFO] Starting training process...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[INFO] Using device: {device}")
    print("[INFO] Creating dataset...")
    try:
        dataset = FruitSegmentationDataset(IMAGE_DIR + "/" +  fruit, MASK_DIR + "/" +  fruit)
        print(f"[INFO] Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return
    print(f"[INFO] Loaded dataset with {len(dataset)} samples")
    print("[INFO] Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    print("[INFO] Initializing Mask R-CNN model...")
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0005)

    model.train()
    for epoch in range(EPOCHS):
        print(f"[INFO] Starting epoch {epoch + 1}/{EPOCHS}...")
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(dataloader):
            #print(f"[BATCH {batch_idx + 1}] Forward pass starting...")
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            #print(f"[BATCH {batch_idx + 1}] Forward pass completed.")
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "mask_rcnn_" + fruit + ".pth")
    print("Model saved to mask_rcnn_" + fruit + ".pth")

if __name__ == '__main__':
    train("Kiwi")