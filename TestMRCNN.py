import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision
import numpy as np
import random
from torchvision.ops import nms

def test(fruit):
    IMAGE_DIR = 'composited'
    MODEL_PATH = 'mask_rcnn_' + fruit + '.pth'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the pre-trained Mask R-CNN model
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Function to run inference on a single image
    def run_inference_on_image(image_path, score_thresh=0.7, iou_thresh=0.2, min_area=10, max_ar=4.0):
        image = Image.open(image_path).convert("RGB")
        original = image.copy()
        image_tensor = F.to_tensor(image).to(DEVICE)

        with torch.no_grad():
            output = model([image_tensor])[0]

        boxes = output['boxes']
        masks = output['masks'] > 0.5
        scores = output['scores']

        keep = scores >= score_thresh
        boxes, masks, scores = boxes[keep], masks[keep], scores[keep]

        keep_nms = nms(boxes, scores, iou_thresh)
        boxes, masks, scores = boxes[keep_nms], masks[keep_nms], scores[keep_nms]

        wh = boxes[:, 2:] - boxes[:, :2]
        areas = wh[:, 0] * wh[:, 1]
        ar = wh[:, 0] / (wh[:, 1] + 1e-6)
        keep_sz = (areas >= min_area) & (ar <= max_ar) & (ar >= 1 / max_ar)
        boxes, masks, scores = boxes[keep_sz], masks[keep_sz], scores[keep_sz]

        count = 0
        plt.figure(figsize=(10, 5))
        plt.title("Detection")
        plt.axis('off')

        for i in range(len(scores)):
            if scores[i] < 0.5:
                continue
            count += 1
            box = boxes[i].cpu().numpy().astype(int)
            mask = masks[i, 0].cpu().numpy()

            plt.subplot(1, 2, 1)
            plt.imshow(np.array(original))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(np.array(original))
            plt.contour(mask, colors='r', linewidths=1)
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                              fill=False, edgecolor='lime', linewidth=2))
            plt.title(f"Detection #{count}")
            plt.axis('off')
            plt.tight_layout()
        plt.show()
        print(f"Total fruits detected: {count}")

    # Limit inference to the specific fruit folder
    fruit_dir = os.path.join(IMAGE_DIR, fruit)
    if not os.path.isdir(fruit_dir):
        print(f"No directory found for fruit: {fruit}")
        return

    all_images = [os.path.join(fruit_dir, f) for f in os.listdir(fruit_dir) if f.endswith('.png')]
    if not all_images:
        print("No images found for inference.")
        return

    test_image = random.choice(all_images)
    print(f"Running inference on: {test_image}")
    run_inference_on_image(test_image)

if __name__ == '__main__':
    for i in range(20) :
        test("Kiwi")
