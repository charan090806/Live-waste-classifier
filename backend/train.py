from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on the waste dataset.")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    print(f"Initializing YOLO configuration for waste classification...")
    
    # Using YOLOv8 nano as base model
    model = YOLO("yolov8n.pt")
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset configuration file not found at {args.data}")
        print("Please structure your dataset with images and labels, and create a data.yaml file.")
        print()
        print("Example data.yaml:")
        print("train: dataset/train/images")
        print("val: dataset/val/images")
        print("\nnames:")
        print("  0: Cardboard")
        print("  1: Food Organics")
        print("  2: Glass")
        print("  3: Metal")
        print("  4: Miscellaneous Trash")
        print("  5: Paper")
        print("  6: Plastic")
        print("  7: Textile Trash")
        print("  8: Vegetation")
        return
        
    print(f"Starting training on {args.data} for {args.epochs} epochs.")
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgz,
        name="waste_model"
    )
    print("Training complete! Copy the best.pt from the runs/detect/waste_model/weights folder to the model/ directory.")

if __name__ == "__main__":
    main()
