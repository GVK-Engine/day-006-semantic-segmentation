# Day 6 — Semantic Segmentation as a ROS2 Node

> **Series 1: Perception | Project 6 of 12**
> MS Robotics & Autonomous Systems Engineering — Arizona State University — Dec 2026

---

## What This Project Does

Days 1–5 put **boxes** around objects. Boxes work for cars and people.
But you cannot put a box around **road**. You cannot box **sky** or **driveable space**.

Those are not objects. They are the **scene itself**.

Semantic segmentation labels **every single pixel** with a class.
465,750 labels from one image. The planning module finally sees the full picture.

```
Raw camera image  →  DeepLabV3 on RTX 4050  →  Per-pixel labels  →  ROS2 topic
```

---

## Results

| Result | Link |
|--------|------|
| Segmentation Frame 0001 — Germany road | [View](https://drive.google.com/file/d/1fZ5yL9L-_kj68j8Iq2-6GZ7ZyMjJwOj5/view?usp=drive_link) |
| Segmentation Frame 0005 — Multiple cars | [View](https://drive.google.com/file/d/1VEoxWGkoKdj86UHfHSIXhiBYd47fA7fL/view?usp=drive_link) |
| Segmentation Frame 0010 — Car + tram | [View](https://drive.google.com/file/d/1OIHWZ5WzelV8fVbY0Cwe2sLo9xjbeaZJ/view?usp=drive_link) |
| Benchmark — FPS across 9 frames | [View](https://drive.google.com/file/d/15EpoTAyKq0ee-sKRnsBlXhGYaco3HI6n/view?usp=drive_link) |
| Evaluation — FPS vs resolution + classes | [View](https://drive.google.com/file/d/1JIVMbQlau73d0mnzJ4JKTmMxhqefpcPY/view?usp=drive_link) |

---

## Benchmark — NVIDIA RTX 4050 GPU

| Metric | Value |
|--------|-------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6.0 GB |
| CUDA | 13.0 — PyTorch 2.6.0+cu124 |
| Avg FPS (stable) | **52.6 FPS** |
| Max FPS | **79.6 FPS** |
| Avg latency | **17ms per frame** |
| Real-time threshold | ✅ EXCEEDED (30 FPS) |
| Frames processed | 20 real KITTI frames |
| ROS2 messages published | 60 total (3 topics × 20 frames) |
| Node uptime | 0.9 seconds |

### FPS vs Image Resolution

| Resolution | FPS | Latency | Status |
|------------|-----|---------|--------|
| 310 × 93 | 104.3 | 9.6ms | ✅ |
| 621 × 187 | 95.7 | 10.4ms | ✅ |
| 931 × 281 | 74.2 | 13.5ms | ✅ |
| **1242 × 375** | **44.6** | **22.4ms** | ✅ Full KITTI |
| 1552 × 468 | 30.1 | 33.2ms | ✅ |

Every resolution exceeds the 30 FPS real-time threshold on RTX 4050.

---

## Class Detection — Real KITTI Germany Road

| Class | Detection Rate | Avg Pixel Coverage |
|-------|---------------|-------------------|
| car | 9 / 9 frames — **100%** | 2.7% |
| train / tram | 8 / 9 frames — 89% | 0.1% |
| person | 3 / 9 frames — 33% | 0.1% |
| bus | 1 / 9 frames — 11% | 0.1% |
| bicycle | 1 / 9 frames — 11% | 0.1% |
| background | 9 / 9 frames — 100% | 96.5% |

> The tram running alongside the German road is correctly classified as **train**.
> Background (96.5%) includes road surface, sky, trees, and buildings —
> COCO does not separate these into road-specific classes.
> A Cityscapes fine-tuned model would label each separately with the same architecture.

---

## Architecture — DeepLabV3 + MobileNetV3

```
INPUT  (1242 × 375 × 3)
   │
   ▼
MobileNetV3 Encoder
   Extracts features at multiple scales
   Shrinks to 78 × 24 (dense feature map)
   │
   ▼
Atrous (Dilated) Convolutions
   Larger receptive field — no resolution loss
   Context: pixels surrounded by road = road
   │
   ▼
DeepLabV3 Decoder
   Upsamples back to 1242 × 375
   Combines multi-scale features
   │
   ▼
OUTPUT  (1242 × 375 × 21)
   21 class scores per pixel
   argmax → class label → color
```

| Property | Value |
|----------|-------|
| Parameters | 11,029,328 |
| Pretrained on | COCO dataset (330k images, 80 classes) |
| Classes used | 21 |
| Training required | None — weights loaded directly |

---

## ROS2 Node Architecture

```
┌─────────────────┐     /camera/image_raw      ┌──────────────────────┐
│   Camera Node   │ ──────────────────────────► │  segmentation_node   │
└─────────────────┘                             │                      │
                                                │  DeepLabV3 on GPU    │
                          /segmentation/mask    │  52.6 FPS avg        │
┌─────────────────┐ ◄─── /segmentation/labels ─│  17ms latency        │
│ Navigation Node │      /segmentation/stats    └──────────────────────┘
└─────────────────┘
```

| Topic | Direction | Type | Content |
|-------|-----------|------|---------|
| `/camera/image_raw` | Subscribe | `sensor_msgs/Image` | Raw BGR camera frame |
| `/segmentation/mask` | Publish | `sensor_msgs/Image` | Colorized RGB mask |
| `/segmentation/labels` | Publish | `sensor_msgs/Image` | Per-pixel class IDs (uint8) |
| `/segmentation/stats` | Publish | `std_msgs/String` | JSON: fps, latency, classes |

To deploy on a real robot:
```bash
# 1. Install ROS2 Humble on Ubuntu 22.04
# 2. Replace SimulatedNode with rclpy.node.Node
# 3. Launch:
ros2 run day006 segmentation_node
```

---

## Key Engineering Findings

**Finding 1 — GPU warmup is real and must be accounted for**

Frame 1 runs at 4.5 FPS (220ms). Frame 2 onwards runs at 33–79 FPS.
The first inference triggers CUDA JIT kernel compilation.
Production systems always run a warmup pass on startup.
I measured and reported it explicitly rather than hiding it.

**Finding 2 — FPS scales predictably with resolution**

Halving resolution (1242→621) gives ~2× FPS gain despite 4× fewer pixels.
GPU operations have fixed overhead per call regardless of image size.
For a system running 8 cameras simultaneously, resolution selection
directly determines whether the full stack meets its latency budget.

**Finding 3 — Node architecture determines real-world value**

Running a model standalone is a tutorial.
Running it as a subscriber-publisher node that integrates
into a robot stack is production engineering.
Any ROS2 robot drops this node in and immediately
has real-time segmentation on `/segmentation/mask`.

---

## Why This Matters to the Industry

| Company | Use of Segmentation |
|---------|-------------------|
| **Waymo** | Driveable surface identification — planner only enters road pixels |
| **Tesla FSD** | Lane marking detection — cameras provide dense pixel data |
| **Boston Dynamics** | Terrain classification per footstep — grass, concrete, void |
| **Amazon Robotics** | Floor/obstacle separation for warehouse navigation |

The fundamental shift: detection answers *"what objects are here?"*
Segmentation answers *"what is everything?"*

Day 8 in this series adds LiDAR depth to the segmentation pipeline.
Every pixel then has a **class** (what is it?) and a **distance** (how far?).
That combination is what a complete AV perception stack requires.

---

## What I Learned

The difference between a **model** and a **system** became clear here.

A model takes an image and returns predictions.
A node takes messages, processes them, and publishes to other nodes.
The model is one function inside the node.
The node is what makes the model useful on a real robot.

Building the topic structure forced me to think about data flow:
what format does the camera publish, what does navigation expect,
how do timestamps keep nodes synchronized.
None of these questions exist in a notebook.
They only appear when you build an actual pipeline.

**Atrous convolutions** were the key architectural insight.
Standard convolutions have a fixed receptive field.
Dilated convolutions look at a wider area without adding parameters.
This is what gives DeepLabV3 context-awareness —
a pixel surrounded by road pixels gets labeled road.
That spatial reasoning is the foundation of scene understanding.

---

## Run It Yourself

```bash
git clone https://github.com/GVK-Engine/day-006-semantic-segmentation
cd day-006-semantic-segmentation
pip install -r requirements.txt
```

```bash
# Full segmentation pipeline on KITTI
py -3.11 segmentation.py

# ROS2 node simulation (20 frames, 3 topics)
py -3.11 ros2_node.py

# FPS vs resolution + class evaluation
py -3.11 evaluate.py
```

Update `KITTI_IMAGE_DIR` in each file to your local KITTI path.
KITTI download (free): https://www.cvlibs.net/datasets/kitti/raw_data.php

---

## Project Structure

```
day-006-semantic-segmentation/
├── model.py              DeepLabV3 loader, predictor, colorizer
├── segmentation.py       Full pipeline — 9 frames, 4-panel visualization
├── ros2_node.py          ROS2 publisher node — 3 topics, 60 messages
├── evaluate.py           FPS vs resolution + class detection benchmark
├── requirements.txt      Python dependencies
└── results/
    ├── segmentation_frame0001.png
    ├── segmentation_frame0005.png
    ├── segmentation_frame0010.png
    ├── segmentation_benchmark.png
    └── evaluation_chart.png
```

---

## Stack

`Python 3.11` `PyTorch 2.6+CUDA12.4` `torchvision` `OpenCV` `NumPy` `Matplotlib` `KITTI`

---

## Series 1 — Perception Progress

| # | Project | Status |
|---|---------|--------|
| P1.1 | LiDAR Obstacle Detection Pipeline | ✅ Complete |
| P1.2 | Stereo Camera Depth Analysis | ✅ Complete |
| P1.3 | PointPillars 3D Object Detector | ✅ Complete |
| P1.4 | Multi-Camera BEV Perception | ✅ Complete |
| P1.5 | Multi-Object Tracking SORT | ✅ Complete |
| P1.6 | Semantic Segmentation as ROS2 Node | ✅ Complete |
