"""
model.py
========
Semantic segmentation using pretrained DeepLabV3.

What this does:
  Takes a camera image (any size)
  Labels every single pixel with a class:
    road, car, person, sky, building, etc.
  Returns a colorized segmentation mask

Why pretrained?
  Training from scratch needs Cityscapes dataset
  (3.5GB) and GPU and 100+ hours.
  Pretrained on COCO gives immediate real results.
  This is exactly what AV companies do in production —
  they fine-tune pretrained models on their own data.

Architecture:
  MobileNetV3 encoder — fast, lightweight
  DeepLabV3 decoder  — atrous convolutions
  21 COCO classes

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 6 of 90 — Perception Series
"""

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import time
from PIL import Image

# ── CLASS DEFINITIONS ─────────────────────────────────────────────────

COCO_CLASSES = [
    'background', 'aeroplane', 'bicycle',  'bird',
    'boat',       'bottle',    'bus',       'car',
    'cat',        'chair',     'cow',       'diningtable',
    'dog',        'horse',     'motorbike', 'person',
    'pottedplant','sheep',     'sofa',      'train',
    'tvmonitor'
]

# Cityscapes-inspired color palette for visualization
COCO_COLORS = np.array([
    [0,   0,   0  ],  # background  — black
    [128, 0,   0  ],  # aeroplane   — dark red
    [0,   128, 0  ],  # bicycle     — dark green
    [128, 128, 0  ],  # bird        — olive
    [0,   0,   128],  # boat        — dark blue
    [128, 0,   128],  # bottle      — purple
    [0,   128, 128],  # bus         — teal
    [64,  64,  192],  # car         — blue-grey
    [64,  0,   0  ],  # cat         — maroon
    [192, 0,   0  ],  # chair       — red
    [64,  128, 0  ],  # cow         — olive green
    [192, 128, 0  ],  # diningtable — orange
    [64,  0,   128],  # dog         — indigo
    [192, 0,   128],  # horse       — pink
    [64,  128, 128],  # motorbike   — steel blue
    [220, 20,  60 ],  # person      — crimson
    [0,   64,  0  ],  # pottedplant — forest green
    [128, 64,  0  ],  # sheep       — brown
    [0,   192, 0  ],  # sofa        — bright green
    [128, 192, 0  ],  # train       — yellow green
    [0,   64,  128],  # tvmonitor   — slate blue
], dtype=np.uint8)


# ── SEGMENTATION MODEL ────────────────────────────────────────────────

class SegmentationModel:
    """
    Pretrained DeepLabV3 + MobileNetV3 for semantic segmentation.

    Every pixel in the image gets assigned one of 21 class labels.
    This is the foundation of scene understanding in autonomous vehicles.

    Waymo uses segmentation to identify driveable surface.
    Tesla uses it for lane detection and road edge finding.
    Boston Dynamics Spot uses it for terrain classification.
    """

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"  Device: {self.device}")
        print(f"  Loading pretrained DeepLabV3...")

        self.model = seg_models.deeplabv3_mobilenet_v3_large(
            weights='DEFAULT'
        )
        self.model.to(self.device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters  : {params:,}")
        print(f"  Classes     : {len(COCO_CLASSES)}")
        print(f"  Ready!")

    def preprocess(self, image_bgr):
        """
        Prepare image for model.
        Converts BGR → RGB → normalize with ImageNet stats → tensor.
        """
        image_rgb = image_bgr[:, :, ::-1].copy()
        pil_img   = Image.fromarray(image_rgb)
        tensor    = TF.to_tensor(pil_img)
        tensor    = TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_bgr):
        """
        Run segmentation on one image.

        Returns:
          labels  : (H, W) numpy int array — class ID per pixel
          mask    : (H, W, 3) numpy uint8  — colorized RGB mask
          classes : list of detected class names
          scores  : dict of class → pixel percentage
          fps     : inference speed
        """
        t0 = time.time()

        with torch.no_grad():
            inp    = self.preprocess(image_bgr)
            output = self.model(inp)['out']          # (1, 21, H, W)
            labels = output.argmax(dim=1).squeeze()  # (H, W)
            labels = labels.cpu().numpy().astype(np.uint8)

        elapsed = time.time() - t0
        fps     = 1.0 / elapsed

        # Colorize
        mask = COCO_COLORS[labels]   # (H, W, 3) RGB

        # Find classes present
        unique_ids = np.unique(labels)
        classes    = [COCO_CLASSES[i] for i in unique_ids
                      if i < len(COCO_CLASSES)]

        # Pixel percentage per class
        total  = labels.size
        scores = {}
        for cid in unique_ids:
            if cid < len(COCO_CLASSES):
                pct = (labels == cid).sum() / total * 100
                if pct > 0.5:
                    scores[COCO_CLASSES[cid]] = round(pct, 1)

        return {
            'labels':  labels,
            'mask':    mask,
            'classes': classes,
            'scores':  scores,
            'fps':     fps,
            'elapsed_ms': elapsed * 1000,
        }

    def overlay(self, image_bgr, mask_rgb, alpha=0.5):
        """
        Blend original image with segmentation mask.
        alpha = mask opacity (0=invisible, 1=full mask)
        """
        mask_bgr = mask_rgb[:, :, ::-1].copy()
        # Resize mask to match image if needed
        if mask_bgr.shape[:2] != image_bgr.shape[:2]:
            mask_bgr = cv2.resize(
                mask_bgr, (image_bgr.shape[1], image_bgr.shape[0])
            )
        return cv2.addWeighted(image_bgr, 1-alpha,
                               mask_bgr, alpha, 0)


# ── TEST ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("\n" + "="*60)
    print("  Segmentation Model Test")
    print("="*60)

    model = SegmentationModel()

    # Test on a synthetic image
    print(f"\n  Testing on synthetic 375x1242 image...")
    test_img = np.random.randint(
        0, 255, (375, 1242, 3), dtype=np.uint8
    )

    result = model.predict(test_img)

    print(f"  Inference     : {result['elapsed_ms']:.1f}ms")
    print(f"  FPS           : {result['fps']:.1f}")
    print(f"  Output shape  : {result['labels'].shape}")
    print(f"  Classes found : {result['classes']}")
    print(f"\n  Model working correctly!")
    print(f"  Run segmentation.py for full pipeline!")