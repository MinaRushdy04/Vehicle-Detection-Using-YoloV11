# YOLO-Focused Feature Ideas

## 1. Upgrade Detector To YOLOv11

Replace the current YOLOv3 Darknet model with a YOLOv11 model. This is the strongest improvement because it directly improves the detection layer while keeping the existing tracking, counting, and logging pipeline.

## 2. Compare YOLO Models

Add a small benchmark mode that runs the same video through different YOLO models and reports:

- detection FPS
- number of detected vehicles
- confidence distribution
- final line-crossing count

This shows understanding of the tradeoff between model speed and detection quality.

## 3. Region Of Interest Detection

Process only a road region instead of the full frame. This reduces false positives and improves speed because YOLO receives less irrelevant image area.

## 4. Vehicle Class Distribution

Use YOLO class labels to report counts by vehicle type:

- car
- motorbike
- bus
- truck

This turns the project from a simple counter into a traffic composition analyzer.

## 5. Confidence-Based Event Review

Flag low-confidence counted vehicles for manual review. This makes the system more explainable because uncertain detections are not silently trusted.

## 6. Detection Heatmap

Use YOLO bounding-box centers to build a heatmap of where vehicles appear most often. This is useful for showing lane density and traffic flow.

## 7. YOLO + OCR Extension

Add license-plate detection/OCR as a second-stage model after YOLO detects a vehicle. This would be an optional advanced extension, not part of the base counter.
