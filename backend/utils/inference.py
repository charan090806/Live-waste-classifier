import os
from ultralytics import YOLO

# Define classes based on the dataset
WASTE_CLASSES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

COCO_TO_WASTE_MAP = {
    'bottle': 'Plastic',
    'wine glass': 'Glass',
    'cup': 'Paper',
    'fork': 'Metal',
    'knife': 'Metal',
    'spoon': 'Metal',
    'bowl': 'Plastic',
    'banana': 'Food Organics',
    'apple': 'Food Organics',
    'sandwich': 'Food Organics',
    'orange': 'Food Organics',
    'broccoli': 'Food Organics',
    'carrot': 'Food Organics',
    'hot dog': 'Food Organics',
    'pizza': 'Food Organics',
    'donut': 'Food Organics',
    'cake': 'Food Organics',
    'potted plant': 'Vegetation',
    'book': 'Paper',
    'scissors': 'Metal',
    'teddy bear': 'Textile Trash',
    'hair drier': 'Plastic',
    'toothbrush': 'Plastic',
    'keyboard': 'Plastic',
    'cell phone': 'Metal',
    'mouse': 'Plastic',
    'remote': 'Plastic',
    'laptop': 'Metal',
    'backpack': 'Textile Trash',
    'umbrella': 'Miscellaneous Trash',
    'handbag': 'Textile Trash',
    'tie': 'Textile Trash',
    'suitcase': 'Miscellaneous Trash',
    'clock': 'Miscellaneous Trash',
    'vase': 'Glass'
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "best.pt")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load model, fallback to yolov8n.pt if not found just to ensure the app doesn't crash
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Loaded custom model from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}. Downloading generic YOLOv8n.pt for placeholder detections...")
        model = YOLO("yolov8n.pt") # fallback for demonstration
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    model = None

def get_predictions(image):
    if model is None:
        return []

    # Run inference
    results = model(image, verbose=False)
    
    predictions = []
    
    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # get box coordinates in (left, top, right, bottom) format
            b = box.xyxy[0].tolist() 
            c = box.cls[0].item()
            conf = box.conf[0].item()
            
            # Map index to class name
            if os.path.exists(MODEL_PATH):
                 # If custom model, map to our classes if possible, or use model names
                 if hasattr(model, 'names') and int(c) in model.names:
                     class_name = model.names[int(c)]
                 else:
                     class_name = WASTE_CLASSES[int(c)] if int(c) < len(WASTE_CLASSES) else f"Class {int(c)}"
            else:
                 # Standard yolo classes mapped from base model
                 coco_class = model.names[int(c)] if hasattr(model, 'names') else str(int(c))
                 if coco_class in COCO_TO_WASTE_MAP:
                     class_name = COCO_TO_WASTE_MAP[coco_class]
                 elif coco_class in ['person', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'airplane', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
                     continue
                 else:
                     class_name = "Miscellaneous Trash"
                 
            # filter out low confidence
            if conf > 0.4:
                # Add friendly percentage strings like "92%" for frontend
                confidence_pct = f"{int(conf * 100)}%"
                predictions.append({
                    "box": [int(x) for x in b],
                    "class": class_name,
                    "confidence": float(f"{conf:.2f}"),
                    "confidence_pct": confidence_pct
                })
            
    return predictions
