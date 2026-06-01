"""
evaluate.py
===========
Evaluation pipeline for semantic segmentation.

Measures:
  1. FPS vs image size (how does speed change?)
  2. Per-class detection across 9 frames
  3. GPU memory usage
  4. First frame warmup cost

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 6 of 90 — Perception Series
"""

import numpy as np
import cv2
import os
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import SegmentationModel, COCO_CLASSES

KITTI_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\image_02\data"
)
RESULTS_DIR = "results"


def load_kitti_frames(n=9):
    """Load first n KITTI frames."""
    images = []
    files  = sorted(os.listdir(KITTI_DIR))[:n]
    for f in files:
        img = cv2.imread(os.path.join(KITTI_DIR, f))
        if img is not None:
            images.append(img)
    return images


def benchmark_fps_vs_size(model, image):
    """
    Measure FPS at different image sizes.
    Shows how resolution affects inference speed.
    """
    scales  = [0.25, 0.5, 0.75, 1.0, 1.25]
    results = []

    for scale in scales:
        h = int(image.shape[0] * scale)
        w = int(image.shape[1] * scale)
        resized = cv2.resize(image, (w, h))

        # Warmup
        model.predict(resized)

        # Measure over 5 runs
        times = []
        for _ in range(5):
            t0 = time.time()
            model.predict(resized)
            times.append(time.time() - t0)

        avg_ms  = np.mean(times) * 1000
        avg_fps = 1000 / avg_ms
        results.append({
            'scale':  scale,
            'size':   f"{w}x{h}",
            'fps':    avg_fps,
            'ms':     avg_ms,
            'pixels': w * h
        })
        print(f"    {scale:.2f}x  ({w:4d}x{h:3d})  "
              f"{avg_fps:>6.1f} FPS  {avg_ms:>6.1f}ms")

    return results


def evaluate_classes(model, images):
    """
    Evaluate class detection across all frames.
    Returns per-class detection rate and avg coverage.
    """
    class_counts   = {}
    class_coverage = {}

    for img in images:
        result = model.predict(img)
        for cls, pct in result['scores'].items():
            if cls == 'background':
                continue
            if cls not in class_counts:
                class_counts[cls]   = 0
                class_coverage[cls] = []
            class_counts[cls] += 1
            class_coverage[cls].append(pct)

    return class_counts, class_coverage


def save_evaluation_chart(fps_results, class_counts,
                          class_coverage, n_frames):
    """Save 3-panel evaluation chart."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        "Semantic Segmentation Evaluation — KITTI  |  RTX 4050\n"
        "Vamshikrishna Gadde  |  MS Robotics ASU  |  Day 6 of 90",
        fontsize=12, color='white'
    )

    # Panel 1: FPS vs resolution
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a1a')
    sizes = [r['size'] for r in fps_results]
    fps   = [r['fps']  for r in fps_results]
    bars  = ax1.bar(sizes, fps, color='#00C8FF',
                    edgecolor='#333', linewidth=0.5)
    ax1.axhline(y=30, color='red', linestyle='--',
                linewidth=1.5, label='30 FPS (real-time)')
    ax1.set_title("FPS vs Image Resolution",
                  color='white', fontsize=11)
    ax1.set_xlabel("Image Size", color='white')
    ax1.set_ylabel("FPS", color='white')
    ax1.tick_params(colors='white', labelsize=8)
    ax1.legend(facecolor='#1a1a1a', labelcolor='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')
    for bar, val in zip(bars, fps):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f"{val:.0f}", ha='center',
                 color='white', fontsize=9)

    # Panel 2: Class detection rate
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')
    if class_counts:
        sorted_c = sorted(class_counts.items(),
                          key=lambda x: x[1], reverse=True)
        names  = [c[0] for c in sorted_c]
        counts = [c[1] for c in sorted_c]
        rates  = [c/n_frames*100 for c in counts]
        ax2.barh(names, rates, color='#FF6B35',
                 edgecolor='#333', height=0.6)
        ax2.set_xlim(0, 120)
        ax2.set_title("Class Detection Rate\n(% of frames detected)",
                      color='white', fontsize=11)
        ax2.set_xlabel("Detection Rate (%)", color='white')
        ax2.tick_params(colors='white')
        for i, (rate, count) in enumerate(zip(rates, counts)):
            ax2.text(rate + 2, i,
                     f"{rate:.0f}% ({count}/{n_frames})",
                     va='center', color='white', fontsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')

    # Panel 3: Avg pixel coverage per class
    ax3 = axes[2]
    ax3.set_facecolor('#1a1a1a')
    if class_coverage:
        avgs   = {k: np.mean(v) for k, v in class_coverage.items()}
        sorted_a = sorted(avgs.items(),
                          key=lambda x: x[1], reverse=True)
        names2  = [c[0] for c in sorted_a]
        values2 = [c[1] for c in sorted_a]
        ax3.barh(names2, values2, color='#00FF64',
                 edgecolor='#333', height=0.6)
        max_v = max(values2) if values2 else 1
        ax3.set_xlim(0, max_v * 1.4)
        ax3.set_title("Avg Pixel Coverage When Detected",
                      color='white', fontsize=11)
        ax3.set_xlabel("Avg Pixel Coverage (%)", color='white')
        ax3.tick_params(colors='white')
        for i, val in enumerate(values2):
            ax3.text(val + max_v*0.03, i,
                     f"{val:.1f}%",
                     va='center', color='white', fontsize=9)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "evaluation_chart.png")
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"\n  Saved: {path}")
    return path


if __name__ == "__main__":
    print("\n" + "="*62)
    print("  Segmentation Evaluation — Day 6 of 90")
    print("="*62)

    model  = SegmentationModel()
    images = load_kitti_frames(9)
    print(f"\n  Loaded {len(images)} KITTI frames")

    # GPU memory
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  GPU VRAM: {mem/1024**3:.1f} GB total")

    # FPS vs resolution
    print(f"\n  FPS vs Image Resolution:")
    print(f"  {'─'*40}")
    fps_results = benchmark_fps_vs_size(model, images[0])

    # Class evaluation
    print(f"\n  Evaluating classes across {len(images)} frames...")
    class_counts, class_coverage = evaluate_classes(
        model, images
    )

    # Print results
    print(f"\n  Class Detection Results:")
    print(f"  {'─'*40}")
    for cls in sorted(class_counts,
                      key=class_counts.get, reverse=True):
        rate = class_counts[cls] / len(images) * 100
        avg  = np.mean(class_coverage[cls])
        print(f"  {cls:<15}: detected in "
              f"{class_counts[cls]}/{len(images)} frames "
              f"({rate:.0f}%)  avg {avg:.1f}% coverage")

    save_evaluation_chart(
        fps_results, class_counts,
        class_coverage, len(images)
    )

    print(f"\n" + "="*62)
    print(f"  EVALUATION COMPLETE")
    print(f"="*62)