"""
segmentation.py
===============
Full semantic segmentation pipeline on real KITTI data.

What this does:
  Loads real KITTI camera images (same road data as Day 1+2)
  Runs DeepLabV3 on RTX 4050 GPU at 43+ FPS
  Creates 4-panel visualization per frame:
    Panel 1: Original KITTI camera image
    Panel 2: Colorized segmentation mask
    Panel 3: Blended overlay (image + mask 50/50)
    Panel 4: Class distribution bar chart
  Benchmarks FPS and latency across 9 frames
  Saves all results for README and LinkedIn

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 6 of 90 — Perception Series
"""

import numpy as np
import cv2
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import SegmentationModel, COCO_CLASSES, COCO_COLORS

# ── PATHS ─────────────────────────────────────────────────────────────

KITTI_IMAGE_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\image_02\data"
)
RESULTS_DIR = "results"


# ── VISUALIZATION ─────────────────────────────────────────────────────

def create_visualization(image_bgr, result, frame_id):
    """
    Create 4-panel visualization for one frame.

    Panel 1: Original camera image
    Panel 2: Pure segmentation mask (colorized)
    Panel 3: Overlay (50% image + 50% mask)
    Panel 4: Class distribution bar chart (all classes)
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle(
        f"Semantic Segmentation — KITTI Frame {frame_id:04d}  "
        f"|  {result['fps']:.1f} FPS  "
        f"|  {result['elapsed_ms']:.0f}ms  "
        f"|  RTX 4050 GPU\n"
        f"Vamshikrishna Gadde  |  MS Robotics ASU  |  Day 6 of 90",
        fontsize=12, color='white', y=0.99
    )

    # ── Panel 1: Original image ───────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Camera Image — KITTI Germany",
                  color='white', fontsize=10, pad=6)
    ax1.axis('off')

    # ── Panel 2: Segmentation mask ────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.imshow(result['mask'])
    ax2.set_title(
        "Semantic Segmentation Mask\n"
        "Every pixel labeled with a class",
        color='white', fontsize=10, pad=6
    )
    ax2.axis('off')

    # Legend for detected non-background classes
    patches = []
    for cls_name in result['classes']:
        if cls_name == 'background':
            continue
        idx   = COCO_CLASSES.index(cls_name)
        color = COCO_COLORS[idx] / 255.0
        pct   = result['scores'].get(cls_name, 0)
        patch = mpatches.Patch(
            color=color,
            label=f"{cls_name} ({pct:.1f}%)"
        )
        patches.append(patch)
    if patches:
        ax2.legend(
            handles=patches, loc='lower right',
            fontsize=8, framealpha=0.7,
            facecolor='#1a1a1a', labelcolor='white',
            edgecolor='#444444'
        )

    # ── Panel 3: Overlay ──────────────────────────────────────────────
    ax3 = axes[1, 0]
    mask_bgr = result['mask'][:, :, ::-1].copy()
    if mask_bgr.shape[:2] != image_bgr.shape[:2]:
        mask_bgr = cv2.resize(
            mask_bgr,
            (image_bgr.shape[1], image_bgr.shape[0])
        )
    overlay = cv2.addWeighted(image_bgr, 0.55, mask_bgr, 0.45, 0)
    ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax3.set_title(
        "Segmentation Overlay — 55% image + 45% mask",
        color='white', fontsize=10, pad=6
    )
    ax3.axis('off')

    # ── Panel 4: Class distribution ───────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor('#1a1a1a')

    # Include ALL detected classes including background
    all_scores = {
        k: v for k, v in result['scores'].items()
        if v > 0.3
    }

    if all_scores:
        sorted_scores = sorted(
            all_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        names  = [s[0] for s in sorted_scores]
        values = [s[1] for s in sorted_scores]
        bar_colors = []
        for n in names:
            if n in COCO_CLASSES:
                c = COCO_COLORS[COCO_CLASSES.index(n)] / 255.0
                bar_colors.append(c)
            else:
                bar_colors.append([0.4, 0.4, 0.4])

        bars = ax4.barh(
            names, values,
            color=bar_colors,
            height=0.55,
            edgecolor='#444444',
            linewidth=0.5
        )

        # Fix axis so labels fit inside
        max_val = max(values)
        ax4.set_xlim(0, max_val * 1.3)

        for bar, val in zip(bars, values):
            ax4.text(
                bar.get_width() + max_val * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va='center', color='white', fontsize=8
            )

        ax4.set_xlabel("Pixel Coverage (%)", color='white',
                       fontsize=9)
        ax4.set_title(
            "Class Distribution — % of image pixels",
            color='white', fontsize=10, pad=6
        )
        ax4.tick_params(colors='white', labelsize=8)
        for spine in ax4.spines.values():
            spine.set_edgecolor('#444444')
    else:
        ax4.text(
            0.5, 0.5,
            "No classes detected\n(all background)",
            ha='center', va='center',
            color='#888888', fontsize=11,
            transform=ax4.transAxes
        )
        ax4.set_title("Class Distribution",
                      color='white', fontsize=10)

    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_DIR,
        f"segmentation_frame{frame_id:04d}.png"
    )
    plt.savefig(path, dpi=130,
                bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    return path


# ── BENCHMARK ─────────────────────────────────────────────────────────

def run_benchmark(model, frames_to_test):
    """
    Run segmentation across multiple frames.
    Skips first frame from FPS average (GPU warmup).
    """
    print(f"\n  Running benchmark on {len(frames_to_test)} frames...")
    print(f"  {'─'*58}")

    all_fps        = []
    all_fps_stable = []   # exclude first frame warmup
    all_classes    = {}
    saved_paths    = []

    for i, frame_id in enumerate(frames_to_test):
        img_path = os.path.join(
            KITTI_IMAGE_DIR,
            f"{frame_id:010d}.png"
        )

        if not os.path.exists(img_path):
            print(f"  Frame {frame_id:04d} — not found, skipping")
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        result = model.predict(image)
        all_fps.append(result['fps'])

        # Skip first frame for stable avg (GPU warmup)
        if i > 0:
            all_fps_stable.append(result['fps'])

        # Accumulate class stats
        for cls, pct in result['scores'].items():
            if cls not in all_classes:
                all_classes[cls] = []
            all_classes[cls].append(pct)

        non_bg = [
            c for c in result['classes']
            if c != 'background'
        ]
        warmup = " [GPU warmup]" if i == 0 else ""
        print(
            f"  Frame {frame_id:04d}: "
            f"{result['fps']:>6.1f} FPS  "
            f"{result['elapsed_ms']:>6.1f}ms  "
            f"Objects: {non_bg}{warmup}"
        )

        # Save first 3 frames
        if len(saved_paths) < 3:
            path = create_visualization(image, result, frame_id)
            saved_paths.append(path)
            print(f"           → Saved: {path}")

    return all_fps, all_fps_stable, all_classes, saved_paths


# ── SUMMARY CHART ─────────────────────────────────────────────────────

def save_summary_chart(all_fps, all_fps_stable, all_classes):
    """
    Save 2-panel benchmark summary:
      Left:  FPS per frame (with warmup shown separately)
      Right: Average class coverage across all frames
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        "Semantic Segmentation Benchmark — KITTI Real Data  |  "
        "RTX 4050 GPU\n"
        "Vamshikrishna Gadde  |  MS Robotics ASU  |  Day 6 of 90",
        fontsize=12, color='white'
    )

    # ── Left: FPS per frame ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a1a')

    bar_colors = [
        '#FF6B35' if i == 0 else '#00C8FF'
        for i in range(len(all_fps))
    ]
    ax1.bar(range(len(all_fps)), all_fps,
            color=bar_colors, alpha=0.9,
            edgecolor='#333333', linewidth=0.5)

    if all_fps_stable:
        avg_stable = np.mean(all_fps_stable)
        ax1.axhline(
            y=avg_stable, color='yellow',
            linestyle='--', linewidth=1.8,
            label=f"Avg (stable): {avg_stable:.1f} FPS"
        )

    ax1.axhline(
        y=30, color='red', linestyle=':',
        linewidth=1.5, label='Real-time threshold (30 FPS)'
    )

    ax1.set_title("Inference Speed — FPS per Frame",
                  color='white', fontsize=11, pad=8)
    ax1.set_xlabel("Frame Index", color='white', fontsize=9)
    ax1.set_ylabel("Frames Per Second", color='white', fontsize=9)
    ax1.tick_params(colors='white')
    ax1.legend(
        facecolor='#1a1a1a', labelcolor='white',
        fontsize=9, edgecolor='#444444'
    )

    # Annotate warmup bar
    if all_fps:
        ax1.text(
            0, all_fps[0] + 0.5,
            "GPU\nwarmup",
            ha='center', color='#FF6B35',
            fontsize=7
        )

    for spine in ax1.spines.values():
        spine.set_edgecolor('#444444')

    # ── Right: Class coverage ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')

    # Include all classes with > 0.3% coverage
    class_avgs = {
        k: np.mean(v) for k, v in all_classes.items()
        if np.mean(v) > 0.3
    }

    if class_avgs:
        sorted_c = sorted(
            class_avgs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        names  = [c[0] for c in sorted_c]
        values = [c[1] for c in sorted_c]
        colors = []
        for n in names:
            if n in COCO_CLASSES:
                c = COCO_COLORS[COCO_CLASSES.index(n)] / 255.0
                colors.append(c)
            else:
                colors.append([0.4, 0.4, 0.4])

        bars = ax2.barh(
            names, values,
            color=colors, height=0.55,
            edgecolor='#444444', linewidth=0.5
        )

        max_val = max(values)
        ax2.set_xlim(0, max_val * 1.3)

        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_width() + max_val * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va='center', color='white', fontsize=9
            )

        ax2.set_title(
            "Avg Class Coverage Across All Frames",
            color='white', fontsize=11, pad=8
        )
        ax2.set_xlabel(
            "Average Pixel Coverage (%)",
            color='white', fontsize=9
        )
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444444')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "segmentation_benchmark.png")
    plt.savefig(path, dpi=130,
                bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"\n  Benchmark chart saved: {path}")
    return path


# ── MAIN ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  Semantic Segmentation Pipeline — Day 6 of 90")
    print("="*62)

    # Load model
    print("\n  Loading DeepLabV3 on GPU...")
    model = SegmentationModel()

    # Check KITTI path
    if not os.path.exists(KITTI_IMAGE_DIR):
        print(f"\n  KITTI not found at: {KITTI_IMAGE_DIR}")
        print(f"  Check your path above.")
    else:
        print(f"\n  KITTI images found!")

        frames = [1, 5, 10, 20, 30, 50, 70, 90, 100]
        all_fps, all_fps_stable, all_classes, paths = \
            run_benchmark(model, frames)

        # Final summary
        avg_stable = np.mean(all_fps_stable) if all_fps_stable else 0
        avg_lat    = 1000 / avg_stable if avg_stable > 0 else 0

        print(f"\n" + "="*62)
        print(f"  BENCHMARK RESULTS — RTX 4050 GPU")
        print(f"  {'='*58}")
        print(f"  Frames tested      : {len(all_fps)}")
        print(f"  Avg FPS (stable)   : {avg_stable:.1f} FPS")
        print(f"  Min FPS            : {min(all_fps):.1f} FPS"
              f"  (frame 1 = GPU warmup)")
        print(f"  Max FPS            : {max(all_fps):.1f} FPS")
        print(f"  Avg latency        : {avg_lat:.0f}ms per frame")
        print(f"  Real-time (30 FPS) : "
              f"{'✅ EXCEEDED' if avg_stable > 30 else '❌ Below'}")
        print(f"\n  Classes detected across all frames:")
        print(f"  {'─'*40}")
        for cls, vals in sorted(
            all_classes.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        ):
            avg = np.mean(vals)
            if avg > 0.3:
                print(f"    {cls:<15}: {avg:>6.1f}% avg pixel coverage")

        save_summary_chart(all_fps, all_fps_stable, all_classes)

        print(f"\n  Images saved to: {RESULTS_DIR}/")
        print(f"  Open segmentation_frame0001.png to see!")
        print(f"\n" + "="*62)
        print(f"  DAY 6 COMPLETE")
        print(f"="*62)