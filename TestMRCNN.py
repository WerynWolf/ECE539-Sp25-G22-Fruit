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

def test(fruit,
         number="",
         score_thresh=0.95,
         iou_thresh=0.2,
         min_area=10,
         max_ar=4.0,
         path=None
         ):

    IMAGE_DIR = 'composited'
    MODEL_PATH = f'models/mask_rcnn_{fruit}{number}.pth'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, 2)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_feats_mask, 256, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    def run_inference_on_image(image_path):
        img = Image.open(image_path).convert("RGB")
        img_t = F.to_tensor(img).to(DEVICE)
        with torch.no_grad():
            out = model([img_t])[0]

        boxes, masks, scores = out['boxes'], out['masks']>0.5, out['scores']
        keep = scores>=score_thresh
        boxes, masks, scores = boxes[keep], masks[keep], scores[keep]
        keep_nms = nms(boxes, scores, iou_thresh)
        boxes, masks, scores = boxes[keep_nms], masks[keep_nms], scores[keep_nms]

        wh = boxes[:,2:]-boxes[:,:2]
        areas = wh[:,0]*wh[:,1]
        ar = wh[:,0]/(wh[:,1]+1e-6)
        keep_sz = (areas>=min_area)&(ar<=max_ar)&(ar>=1/max_ar)
        boxes, masks, scores = boxes[keep_sz], masks[keep_sz], scores[keep_sz]

        plt.figure(figsize=(10,5))
        plt.title("Detection")
        plt.axis('off')
        count = 0
        for i in range(len(scores)):
            if scores[i]<0.5: continue
            count+=1
            box = boxes[i].cpu().numpy().astype(int)
            mask = masks[i,0].cpu().numpy()

            plt.subplot(1,2,1)
            plt.imshow(np.array(img)); plt.axis('off'); plt.title("Original")

            plt.subplot(1,2,2)
            plt.imshow(np.array(img))
            plt.contour(mask, colors='r', linewidths=1)
            plt.gca().add_patch(plt.Rectangle(
                (box[0],box[1]), box[2]-box[0], box[3]-box[1],
                fill=False, edgecolor='lime', linewidth=2))
            plt.axis('off'); plt.title(f"Detection #{count}")
            plt.tight_layout()

        plt.show()
        print(f"Total fruits detected: {count}")

    if path:
        if not os.path.isfile(path):
            print(f"Image not found: {path}")
            return
        test_image = path
    else:
        fruit_dir = os.path.join(IMAGE_DIR, fruit)
        if not os.path.isdir(fruit_dir):
            print(f"No directory for fruit: {fruit}")
            return
        imgs = [os.path.join(fruit_dir,f) for f in os.listdir(fruit_dir) if f.endswith('.png')]
        if not imgs:
            print("No images found for inference.")
            return
        test_image = random.choice(imgs)

    print(f"Running inference on: {test_image}")
    run_inference_on_image(test_image)

def naturalTest(fruit,
         runs=1,
         number="",
         score_thresh=0.95,
         iou_thresh=0.2,
         min_area=10,
         max_ar=4.0):

    IMAGE_DIR = 'test'
    MODEL_PATH = f'models/mask_rcnn_{fruit}{number}.pth'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, 2)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_feats_mask, 256, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    def run_inference_on_image(image_path):
        img = Image.open(image_path).convert("RGB")
        img_t = F.to_tensor(img).to(DEVICE)
        with torch.no_grad():
            out = model([img_t])[0]

        boxes, masks, scores = out['boxes'], out['masks']>0.5, out['scores']
        keep1 = scores >= score_thresh
        boxes, masks, scores = boxes[keep1], masks[keep1], scores[keep1]

        wh = boxes[:,2:] - boxes[:,:2]
        areas = wh[:,0]*wh[:,1]
        ar = wh[:,0]/(wh[:,1]+1e-6)
        keep2 = (areas >= min_area) & (ar <= max_ar) & (ar >= 1/max_ar)
        boxes, masks, scores = boxes[keep2], masks[keep2], scores[keep2]

        boxes_pre, masks_pre, scores_pre = boxes, masks, scores

        keep_n = nms(boxes, scores, iou_thresh)
        boxes_post, masks_post, scores_post = boxes[keep_n], masks[keep_n], scores[keep_n]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, (bxs, mks, title) in zip(axes,
                                         [(boxes_pre, masks_pre, "Before NMS"),
                                          (boxes_post, masks_post, "After NMS")]):
            ax.imshow(np.array(img))
            ax.axis('off')
            ax.set_title(title)
            for i in range(len(bxs)):
                box = bxs[i].cpu().numpy().astype(int)
                mask = mks[i,0].cpu().numpy()
                ax.contour(mask, colors='r', linewidths=1)
                ax.add_patch(plt.Rectangle(
                    (box[0], box[1]),
                    box[2]-box[0], box[3]-box[1],
                    fill=False, edgecolor='lime', linewidth=2
                ))
        plt.tight_layout()
        plt.show()

        print(f"Detections: {len(scores_pre)} → {len(scores_post)} after NMS")

    fruit_dir = os.path.join(IMAGE_DIR, fruit)
    if not os.path.isdir(fruit_dir):
        raise FileNotFoundError(f"No folder: {fruit_dir}")
    imgs = [os.path.join(fruit_dir, f)
            for f in os.listdir(fruit_dir) if f.lower().endswith('.jpg')]
    if not imgs:
        raise FileNotFoundError(f"No .png images in {fruit_dir}")

    for i in range(runs):
        img_path = random.choice(imgs)
        print(f"\nRun {i+1}/{runs} — {img_path}")
        run_inference_on_image(img_path)

def test_params(
        fruit,
        runs=3,
        score_threshs=(0.5, 0.7, 0.9),
        iou_threshs=(0.1, 0.3, 0.5),
        number="",
        min_area=10,
        max_ar=4.0
):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    MODEL_PATH = f"models/mask_rcnn_{fruit}{number}.pth"

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, 2)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_feats_mask, 256, 2
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # pick images
    img_paths = get_random_images(fruit, runs)

    for idx, img_path in enumerate(img_paths, start=1):
        img = Image.open(img_path).convert("RGB")
        img_t = F.to_tensor(img).to(DEVICE)
        with torch.no_grad():
            out = model([img_t])[0]
        boxes0, masks0, scores0 = out['boxes'], out['masks'] > 0.5, out['scores']

        # apply size & aspect filters once
        wh = boxes0[:, 2:] - boxes0[:, :2]
        areas = wh[:, 0] * wh[:, 1]
        ar = wh[:, 0] / (wh[:, 1] + 1e-6)
        filt = (areas >= min_area) & (ar <= max_ar) & (ar >= 1 / max_ar)
        boxes0, masks0, scores0 = boxes0[filt], masks0[filt], scores0[filt]

        # set up grid
        R, C = len(score_threshs), len(iou_threshs)
        fig, axes = plt.subplots(R, C, figsize=(4 * C, 4 * R))
        fig.suptitle(f"Image {idx}/{runs}: {os.path.basename(img_path)}", y=1.02)

        for i, st in enumerate(score_threshs):
            for j, it in enumerate(iou_threshs):
                ax = axes[i][j] if R > 1 and C > 1 else (axes[j] if R == 1 else axes[i])
                ax.imshow(np.array(img))
                ax.set_title(f"s>={st}, i<={it}")
                ax.axis('off')

                keep_s = scores0 >= st
                bxs, mks, scs = boxes0[keep_s], masks0[keep_s], scores0[keep_s]

                keep_n = nms(bxs, scs, it)
                bxs_n, mks_n = bxs[keep_n], mks[keep_n]

                for k in range(len(bxs_n)):
                    box = bxs_n[k].cpu().numpy().astype(int)
                    mask = mks_n[k, 0].cpu().numpy()
                    ax.contour(mask, colors='r', linewidths=1)
                    ax.add_patch(plt.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0], box[3] - box[1],
                        fill=False, edgecolor='lime', linewidth=2
                    ))

        plt.tight_layout()
        plt.show()

def get_random_images(fruit, count, test_dir='test', extensions=('.png', '.jpg', '.jpeg')):
    folder = os.path.join(test_dir, fruit)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"No such directory: {folder}")

    all_imgs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]
    if not all_imgs:
        raise FileNotFoundError(f"No images found in: {folder}")

    if count >= len(all_imgs):
        random.shuffle(all_imgs)
        return all_imgs

    return random.sample(all_imgs, count)


if __name__ == '__main__':
    #test("Apple", "50", 0.7, 0.2, 10, 4, path="composited/Apple/comp_Apple_0000_mult_r0_273_3_r1_208_4_r0_191_8.png")
    #naturalTest("Limes", 20, "20",0.7, 0.2, 10, 4)

    test_params("Cherry",20,number="20")




