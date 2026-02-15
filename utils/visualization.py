"""Visualization utilities for face detection."""

import cv2
import time
import numpy as np
from typing import Tuple


def draw_detections(frame: np.ndarray, results, thickness: int = 2, show_conf: bool = True) -> np.ndarray:
    """
    Draw bounding boxes and labels on the frame.
    
    Args:
        frame: Input image frame
        results: YOLO detection results
        thickness: Line thickness for bounding boxes
        show_conf: Whether to show confidence scores
        
    Returns:
        Frame with drawn detections
    """
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label with confidence
            if show_conf:
                label = f'Face: {conf:.2f}'
                
                # Calculate label size for background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
    
    return frame


def calculate_fps(prev_time: float) -> Tuple[float, float]:
    """
    Calculate frames per second.
    
    Args:
        prev_time: Previous timestamp
        
    Returns:
        Tuple of (fps, current_time)
    """
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
    return fps, current_time


def put_fps_text(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Overlay FPS text on the frame.
    
    Args:
        frame: Input image frame
        fps: Frames per second value
        
    Returns:
        Frame with FPS text
    """
    fps_text = f'FPS: {fps:.1f}'
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (10, 10),
        (20 + text_width, 20 + text_height + baseline),
        (0, 0, 0),
        -1
    )
    
    # Draw FPS text
    cv2.putText(
        frame,
        fps_text,
        (15, 15 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    return frame
