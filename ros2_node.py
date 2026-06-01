"""
ros2_node.py
============
Semantic Segmentation as a ROS2 Publisher Node.

In production this node runs inside a real ROS2 environment.
On Windows (without ROS2 installed) it simulates the
node behavior with identical code structure.

ROS2 Pipeline:
  /camera/image_raw  → [This Node] → /segmentation/mask
                                   → /segmentation/labels
                                   → /segmentation/stats

Real robot integration:
  1. Install ROS2 Humble on Ubuntu
  2. Replace SimulatedNode with rclpy.node.Node
  3. Replace self.publish() with self.publisher_.publish()
  4. Run: ros2 run day006 segmentation_node

The architecture, topic names, message types,
and callback structure are production-ready.

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 6 of 90 — Perception Series
"""

import numpy as np
import cv2
import os
import time
import json
from model import SegmentationModel, COCO_CLASSES

# ── SIMULATED ROS2 TYPES ──────────────────────────────────────────────
# On real ROS2 these come from:
#   from sensor_msgs.msg import Image
#   from std_msgs.msg import String
#   import rclpy
#   from rclpy.node import Node

class ImageMessage:
    """Simulates sensor_msgs/Image ROS2 message."""
    def __init__(self, data, height, width,
                 encoding, timestamp):
        self.data      = data
        self.height    = height
        self.width     = width
        self.encoding  = encoding
        self.timestamp = timestamp


class SegmentationMessage:
    """Simulates custom segmentation ROS2 message."""
    def __init__(self, mask, labels, classes,
                 fps, latency_ms, timestamp):
        self.mask       = mask       # (H,W,3) colorized
        self.labels     = labels     # (H,W) class IDs
        self.classes    = classes    # list of class names
        self.fps        = fps
        self.latency_ms = latency_ms
        self.timestamp  = timestamp


class SimulatedPublisher:
    """
    Simulates rclpy Publisher.
    On real ROS2: rclpy.create_publisher(Image, topic, 10)
    """
    def __init__(self, topic, msg_count=0):
        self.topic     = topic
        self.msg_count = msg_count

    def publish(self, message):
        self.msg_count += 1


class SimulatedSubscriber:
    """
    Simulates rclpy Subscriber.
    On real ROS2: rclpy.create_subscription(Image, topic, cb, 10)
    """
    def __init__(self, topic, callback):
        self.topic    = topic
        self.callback = callback


# ── SEGMENTATION NODE ─────────────────────────────────────────────────

class SegmentationNode:
    """
    Semantic Segmentation ROS2 Node.

    Subscribes to:
      /camera/image_raw     (sensor_msgs/Image)

    Publishes to:
      /segmentation/mask    (sensor_msgs/Image)
        Colorized RGB segmentation mask

      /segmentation/labels  (sensor_msgs/Image)
        Per-pixel class ID map (uint8)

      /segmentation/stats   (std_msgs/String)
        JSON: fps, latency, classes, coverage

    This is the standard structure for any
    perception node in a production robot stack.
    The navigation node subscribes to /segmentation/mask
    and uses it to identify driveable regions.
    """

    NODE_NAME = "segmentation_node"

    def __init__(self):
        print(f"\n  Initializing {self.NODE_NAME}...")

        # Load segmentation model
        self.model = SegmentationModel()

        # Publishers
        self.pub_mask   = SimulatedPublisher(
            "/segmentation/mask"
        )
        self.pub_labels = SimulatedPublisher(
            "/segmentation/labels"
        )
        self.pub_stats  = SimulatedPublisher(
            "/segmentation/stats"
        )

        # Subscriber
        self.sub_camera = SimulatedSubscriber(
            "/camera/image_raw",
            self.image_callback
        )

        # Node state
        self.frame_count = 0
        self.fps_history = []
        self.start_time  = time.time()

        print(f"  Node: {self.NODE_NAME}")
        print(f"  Sub : /camera/image_raw")
        print(f"  Pub : /segmentation/mask")
        print(f"  Pub : /segmentation/labels")
        print(f"  Pub : /segmentation/stats")
        print(f"  Node ready!")

    def image_callback(self, msg: ImageMessage):
        """
        Called every time a new camera frame arrives.

        On real ROS2 this is triggered automatically
        when the camera node publishes a new image.
        The ROS2 executor handles the threading.

        Steps:
          1. Convert ROS2 image message to numpy array
          2. Run segmentation model
          3. Package results into messages
          4. Publish to all output topics
        """
        # Step 1: Convert message to numpy
        image = np.frombuffer(
            msg.data, dtype=np.uint8
        ).reshape(msg.height, msg.width, 3)

        # Step 2: Run segmentation
        result = self.model.predict(image)
        self.frame_count += 1
        self.fps_history.append(result['fps'])

        # Step 3: Package mask message
        mask_msg = ImageMessage(
            data      = result['mask'].tobytes(),
            height    = result['mask'].shape[0],
            width     = result['mask'].shape[1],
            encoding  = "rgb8",
            timestamp = time.time()
        )

        # Step 4: Package labels message
        labels_msg = ImageMessage(
            data      = result['labels'].tobytes(),
            height    = result['labels'].shape[0],
            width     = result['labels'].shape[1],
            encoding  = "mono8",
            timestamp = time.time()
        )

        # Package stats message (JSON string)
        stats = {
            "frame":      self.frame_count,
            "fps":        round(result['fps'], 1),
            "latency_ms": round(result['elapsed_ms'], 1),
            "classes":    result['classes'],
            "coverage":   result['scores'],
            "timestamp":  time.time()
        }
        stats_msg = json.dumps(stats)

        # Publish all topics
        self.pub_mask.publish(mask_msg)
        self.pub_labels.publish(labels_msg)
        self.pub_stats.publish(stats_msg)

        return result

    def get_stats(self):
        """Return node statistics summary."""
        uptime = time.time() - self.start_time
        avg_fps = np.mean(self.fps_history) \
                  if self.fps_history else 0
        return {
            "node":        self.NODE_NAME,
            "frames":      self.frame_count,
            "uptime_s":    round(uptime, 1),
            "avg_fps":     round(avg_fps, 1),
            "msgs_pub":    self.pub_mask.msg_count,
        }


# ── RUN NODE ──────────────────────────────────────────────────────────

def run_node():
    """
    Spin the segmentation node on real KITTI frames.

    On real ROS2 this would be:
      rclpy.init()
      node = SegmentationNode()
      rclpy.spin(node)
      rclpy.shutdown()
    """
    KITTI_DIR = (
        r"C:\Users\vamsh\Downloads\kitti"
        r"\2011_09_26_drive_0001_sync"
        r"\2011_09_26"
        r"\2011_09_26_drive_0001_sync"
        r"\image_02\data"
    )

    print("\n" + "="*62)
    print("  Segmentation ROS2 Node — Simulation Mode")
    print("  Day 6 of 90 — Perception Series")
    print("="*62)

    # Initialize node
    node = SegmentationNode()

    # Get image frames
    if not os.path.exists(KITTI_DIR):
        print(f"\n  KITTI not found: {KITTI_DIR}")
        return

    images = sorted([
        f for f in os.listdir(KITTI_DIR)
        if f.endswith('.png')
    ])[:20]  # Process 20 frames

    print(f"\n  Spinning node on {len(images)} frames...")
    print(f"  {'─'*58}")

    for i, fname in enumerate(images):
        img_path = os.path.join(KITTI_DIR, fname)
        image    = cv2.imread(img_path)
        if image is None:
            continue

        # Simulate camera publishing an image
        msg = ImageMessage(
            data      = image.tobytes(),
            height    = image.shape[0],
            width     = image.shape[1],
            encoding  = "bgr8",
            timestamp = time.time()
        )

        # Node processes the message
        result = node.image_callback(msg)

        non_bg = [
            c for c in result['classes']
            if c != 'background'
        ]
        print(
            f"  [{i+1:02d}/20] "
            f"Frame {fname[:10]}  "
            f"{result['fps']:>6.1f} FPS  "
            f"{result['elapsed_ms']:>5.1f}ms  "
            f"→ Published 3 topics  "
            f"Objects: {non_bg}"
        )

    # Final stats
    stats = node.get_stats()
    print(f"\n" + "="*62)
    print(f"  NODE STATISTICS")
    print(f"  {'='*58}")
    print(f"  Node name     : {stats['node']}")
    print(f"  Frames processed: {stats['frames']}")
    print(f"  Uptime        : {stats['uptime_s']}s")
    print(f"  Avg FPS       : {stats['avg_fps']} FPS")
    print(f"  Messages pub  : {stats['msgs_pub']} "
          f"(mask topic only)")
    print(f"  Total msgs    : {stats['msgs_pub'] * 3} "
          f"(3 topics × {stats['frames']} frames)")
    print(f"\n  Topics active:")
    print(f"    /camera/image_raw  → [subscribed]")
    print(f"    /segmentation/mask → "
          f"[{node.pub_mask.msg_count} messages published]")
    print(f"    /segmentation/labels → "
          f"[{node.pub_labels.msg_count} messages published]")
    print(f"    /segmentation/stats  → "
          f"[{node.pub_stats.msg_count} messages published]")
    print(f"\n  To deploy on real ROS2:")
    print(f"    1. Install ROS2 Humble on Ubuntu 22.04")
    print(f"    2. Replace SimulatedNode with rclpy.node.Node")
    print(f"    3. ros2 run day006 segmentation_node")
    print(f"="*62)


if __name__ == "__main__":
    run_node()