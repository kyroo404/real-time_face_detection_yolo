"""
Real-Time Face Detection using YOLOv26

This script performs real-time face detection using Ultralytics YOLOv26.
It supports both webcam and video file input, with configurable parameters.

NOTE: For best results, use a YOLOv26 model fine-tuned on face detection datasets
(e.g., WIDER FACE). The default pretrained model detects "person" class from COCO,
not faces specifically. Use train.py to fine-tune on a face dataset.

Usage:
    python detect_faces.py                                    # Webcam with defaults
    python detect_faces.py --source video.mp4                 # Video file
    python detect_faces.py --conf 0.5 --model yolo26s.pt      # Custom settings
    python detect_faces.py --save --output output/result.mp4  # Save output
"""

import argparse
import cv2
import sys
import os
from pathlib import Path
from ultralytics import YOLO
from utils.visualization import draw_detections, calculate_fps, put_fps_text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time face detection using YOLOv26',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: 0 for webcam, or path to video file (default: 0)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.40,
        help='Confidence threshold for detections (default: 0.40)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolo26n.pt',
        help='YOLOv26 model variant: yolo26n/s/m/l/x.pt (default: yolo26n.pt)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save output video to disk'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/output.mp4',
        help='Output video path (default: output/output.mp4)'
    )
    
    parser.add_argument(
        '--show-fps',
        action='store_true',
        default=True,
        help='Display FPS on screen (default: True)'
    )
    
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Bounding box line thickness (default: 2)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Inference image size (default: 640)'
    )
    
    return parser.parse_args()


def initialize_video_source(source):
    """
    Initialize video capture from webcam or file.
    
    Args:
        source: Video source (0 for webcam or file path)
        
    Returns:
        cv2.VideoCapture object
        
    Raises:
        RuntimeError: If video source cannot be opened
    """
    # Convert source to int if it's a numeric string (webcam index)
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        if isinstance(source, int):
            raise RuntimeError(f"Failed to open webcam (index {source})")
        else:
            raise RuntimeError(f"Failed to open video file: {source}")
    
    return cap


def initialize_video_writer(cap, output_path, fps=30.0):
    """
    Initialize video writer for saving output.
    
    Args:
        cap: VideoCapture object to get frame properties
        output_path: Path to save output video
        fps: Frames per second for output video
        
    Returns:
        cv2.VideoWriter object
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    return writer


def main():
    """Main function for real-time face detection."""
    args = parse_args()
    
    print("=" * 60)
    print("Real-Time Face Detection with YOLOv26")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Image size: {args.imgsz}")
    print("=" * 60)
    
    # Load YOLOv26 model
    try:
        print(f"\nLoading model '{args.model}'...")
        model = YOLO(args.model)
        print("Model loaded successfully!")
        print("\nNOTE: For face-specific detection, use a model fine-tuned on")
        print("face datasets (e.g., WIDER FACE). The default COCO model detects")
        print("'person' class. Use train.py to fine-tune on face data.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize video source
    try:
        print(f"\nInitializing video source...")
        cap = initialize_video_source(args.source)
        print("Video source initialized successfully!")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Initialize video writer if saving
    writer = None
    if args.save:
        writer = initialize_video_writer(cap, args.output)
        print(f"Output will be saved to: {args.output}")
    
    print("\nStarting detection... Press 'q' to quit.\n")
    
    # Detection loop
    prev_time = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream or failed to read frame.")
                break
            
            # Run YOLOv26 inference
            # NOTE: YOLOv26 is NMS-free, so no separate NMS post-processing needed
            # For person detection (default COCO model), use classes=[0]
            # For face detection with fine-tuned model, use classes=[0] or omit
            results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            
            # Draw detections
            frame = draw_detections(frame, results, thickness=args.thickness)
            
            # Calculate and display FPS
            if args.show_fps:
                fps, prev_time = calculate_fps(prev_time)
                frame = put_fps_text(frame, fps)
            
            # Save frame if requested
            if writer is not None:
                writer.write(frame)
            
            # Display frame
            cv2.imshow('YOLOv26 Face Detection', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError during detection: {e}")
    finally:
        # Cleanup
        print(f"\nProcessed {frame_count} frames.")
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Output saved to: {args.output}")
        cv2.destroyAllWindows()
        print("Resources released. Exiting.")


if __name__ == '__main__':
    main()
