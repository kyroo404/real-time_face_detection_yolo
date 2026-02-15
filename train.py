"""
Fine-tuning script for YOLOv26 face detection.

This script fine-tunes a YOLOv26 model on face detection datasets to enable
accurate face-specific detection instead of general person detection.

Supported datasets:
    - WIDER FACE (with YOLO format conversion)
    - Custom face datasets in YOLO format

Dataset structure (YOLO format):
    datasets/face_detection/
    ├── images/
    │   ├── train/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── val/
    │       ├── img1.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── img1.txt
        │   ├── img2.txt
        │   └── ...
        └── val/
            ├── img1.txt
            └── ...

Label format (YOLO):
    Each .txt file contains bounding boxes for one image:
    <class_id> <x_center> <y_center> <width> <height>
    
    All values are normalized to [0, 1]:
    - class_id: 0 (for 'face')
    - x_center, y_center: center of bounding box
    - width, height: box dimensions

Usage:
    # Train with default settings
    python train.py --data data/face_dataset.yaml
    
    # Train with custom settings
    python train.py --data data/face_dataset.yaml --epochs 100 --batch 16 --imgsz 640
    
    # Train with larger model
    python train.py --data data/face_dataset.yaml --model yolo26m.pt
    
    # Resume training
    python train.py --data data/face_dataset.yaml --resume runs/detect/train/weights/last.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv26 for face detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file (e.g., data/face_dataset.yaml)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolo26n.pt',
        help='Base YOLOv26 model: yolo26n/s/m/l/x.pt (default: yolo26n.pt)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Training image size (default: 640)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='face_detection',
        help='Experiment name for saving results (default: face_detection)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint (path to .pt file)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use: cuda:0, cpu, or mps (default: auto-detect)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers (default: 8)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (default: 50)'
    )
    
    return parser.parse_args()


def validate_data_file(data_path):
    """
    Validate that the dataset YAML file exists.
    
    Args:
        data_path: Path to dataset YAML file
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found: {data_path}\n"
            f"Please create a YAML file with dataset paths and class names.\n"
            f"See data/face_dataset.yaml for an example."
        )


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 70)
    print("YOLOv26 Face Detection Training")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Experiment name: {args.name}")
    print("=" * 70)
    
    # Validate data file
    try:
        validate_data_file(args.data)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    # Load model
    try:
        if args.resume:
            print(f"\nResuming training from: {args.resume}")
            model = YOLO(args.resume)
        else:
            print(f"\nLoading base model: {args.model}")
            model = YOLO(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nError loading model: {e}")
        return
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    print("Note: YOLOv26 uses MuSGD optimizer and Progressive Loss Balancing by default.")
    print("-" * 70)
    
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            device=args.device if args.device else None,
            workers=args.workers,
            patience=args.patience,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            plots=True,
            val=True
        )
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        print(f"Results saved to: runs/detect/{args.name}/")
        print(f"Best model: runs/detect/{args.name}/weights/best.pt")
        print(f"Last model: runs/detect/{args.name}/weights/last.pt")
        print("\nTo use the trained model for detection:")
        print(f"python detect_faces.py --model runs/detect/{args.name}/weights/best.pt")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Training interrupted or failed.")


if __name__ == '__main__':
    main()
