from ultralytics import YOLO
from ultralytics import SAM

# Load the YOLO model
yolo_model = YOLO("Trained YOLO11 Model/best.pt")

# Run batched inference on a list of images
results = yolo_model("Data/sample3.jpg")

# Load the SAM model
sam_model = SAM("sam2_b.pt")

for result in results:
    class_ids = result.boxes.cls.int().tolist()
    if len(class_ids):
        boxes = result.boxes.xyxy # Boxes object for bbox outputs
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=True, device="cpu")