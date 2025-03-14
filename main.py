# %%
#pip install ultralytics opencv-python tqdm

# %%
from ultralytics import YOLO
import os

# Define dataset path
dataset_path = "/ML/ubl vs non-ubl/data_dir"

# Check if directories exist
assert os.path.exists(f"{dataset_path}/train/images"), "Train images folder missing!"
assert os.path.exists(f"{dataset_path}/train/labels"), "Train labels folder missing!"
assert os.path.exists(f"{dataset_path}/valid/images"), "Validation images folder missing!"
assert os.path.exists(f"{dataset_path}/valid/labels"), "Validation labels folder missing!"
assert os.path.exists(f"{dataset_path}/test/images"), "Test images folder missing!"

print("✅ Dataset structure verified.")

# %%
data_yaml_path = f"{dataset_path}/data.yaml"

# Correcting paths inside data.yaml
with open(data_yaml_path, "w") as f:
    f.write(f"""
train: {dataset_path}/train/images
val: {dataset_path}/valid/images
test: {dataset_path}/test/images

nc: 2
names: ['non-ubl','ubl']
    """)

print("✅ data.yaml correctly updated.")

# %%
import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.device_count())  # Should return the number of GPUs available
print(torch.version.cuda)  # Check installed CUDA version

# %%
# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # Using YOLOv8-Small for best speed-accuracy tradeoff

# Train the model
model.train(
    data=f"{dataset_path}/data.yaml",  # Path to updated data.yaml
    epochs=200,                         # Number of training epochs
    batch=16,                           # Batch size (adjust based on your system)
    imgsz=640,                          # Input image size
    device="cuda",
    augment=True                        # Use GPU (if available) for training
)

print("✅ Training completed.")
# Save the trained model
model.save("ublVSnon_ublModel_v1.pt")

# %%
# Validate the trained model
model.val()

print("✅ Validation completed.")

# %%
model.predict(source="/ML/ubl vs non-ubl/data_dir/test/images", save=True, conf=0.5)

# %%
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")  # Path to your trained model

# Define test image path
image_path = "/ML/ubl vs non-ubl/data_dir/valid/images/sh9_jpeg.rf.7c10e0d17ffbfad034c966a5afa8d6f7.jpg"  # Change this to your actual test image

# Run inference
results = model(image_path)

# Process and visualize the results
for result in results:
    # Load image with OpenCV
    img = cv2.imread(image_path)

    # Loop through detected objects
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class ID

        # Get class name from YOLO model
        class_name = model.names[cls]

        # Define a color map for your classes
        color_map = {
            0: (0, 0, 255),    # Green for first class
            1: (0, 255, 0)     # Red for second class
            }
        # Get the color for this class
        box_color = color_map.get(cls, (0, 255, 0))  # Default to green if class not in map
        
        # Draw bounding box with class-specific color
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1)
        cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 2)

    # Convert BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_path = "/ML/ubl vs non-ubl/output/output_image.jpg"  # You can change the path and filename as needed
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")

    # Show image with bounding boxes
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

# %%
# Run validation to see performance metrics
metrics = model.val()

print("📊 Model Performance:")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")


