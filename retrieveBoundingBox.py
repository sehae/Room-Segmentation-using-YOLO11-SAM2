from ultralytics import YOLO
from ultralytics import SAM

# Load the YOLO model
yolo_model = YOLO("Trained YOLO11 Model/best.pt")

# Run batched inference on a list of images
results = yolo_model("Data/sample4.jpg")

# Load the SAM model
sam_model = SAM("sam2_b.pt")

for result in results:
    class_ids = result.boxes.cls.int().tolist()
    if len(class_ids):
        # Extract bounding box coordinates in tensor format
        boxes = result.boxes.xyxy  # Tensor format: [x_min, y_min, x_max, y_max]

        # Convert tensor bounding boxes to a human-readable format
        human_readable_boxes = boxes.tolist()
        print("Human-readable bounding boxes:", human_readable_boxes)

        # Process with SAM using the bounding boxes
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=True, device="cpu")

def get_bounding_box_coordinates(image_width, image_height, x_center, y_center, width, height):
    # Calculate the coordinates of the bounding box
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    half_width = width * image_width / 2
    half_height = height * image_height / 2

    #calculate the coordinates of the bounding box
    x_min = int(x_center_pixel - half_width)
    y_min = int(y_center_pixel - half_height)
    x_max = int(x_center_pixel + half_width)
    y_max = int(y_center_pixel + half_height)

    return x_min, y_min, x_max, y_max