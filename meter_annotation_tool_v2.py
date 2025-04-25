import os
import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import easyocr
import numpy as np
from tqdm import tqdm  # For progress tracking
import threading  # Added this import

# Initialize variables
image_folder = ""
output_excel = "results.xlsx"
annotations = []
current_index = 0
# Directories for saving original and YOLO-ed images
original_dir = "original_images"
yoloed_dir = "yoloed_images"
# Ensure directories exist
os.makedirs(original_dir, exist_ok=True)
os.makedirs(yoloed_dir, exist_ok=True)
# Load the trained YOLO model
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"YOLO model not found at {model_path}. Ensure the model is in the correct location.")
model = YOLO(model_path)
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])
# Global counter for cropped image filenames
global_crop_counter = 0

# Function to validate an image file
def is_valid_image(image_path):
    try:
        Image.open(image_path)
        return True
    except IOError:
        return False

# Function to run YOLO inference on a single image
def detect_objects(image_path):
    if not is_valid_image(image_path):
        st.error(f"Invalid image: {image_path}")
        return []
    try:
        results = model.predict(source=image_path, save=False, conf=0.25)
        detections = []
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2))
        return detections
    except Exception as e:
        st.error(f"Error processing {image_path}: {e}")
        return []

# Function to crop the image using YOLO bounding boxes
def crop_image(image_path, detections):
    global global_crop_counter
    image = Image.open(image_path)
    cropped_images = []
    for bbox in detections:
        x1, y1, x2, y2 = bbox
        cropped_image = image.crop((x1, y1, x2, y2))
        # Save the cropped image with a unique filename
        global_crop_counter += 1
        cropped_filename = f"crop_{global_crop_counter}.jpg"
        cropped_path = os.path.join(yoloed_dir, cropped_filename)
        cropped_image.save(cropped_path)
        cropped_images.append(cropped_image)
    return cropped_images

# Function to perform OCR on a cropped image
def perform_ocr(image):
    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    # Perform OCR using EasyOCR
    result = reader.readtext(image_np, detail=0)
    # Extract numbers only from the OCR result
    numbers = ''.join(filter(str.isdigit, ''.join(result)))
    return numbers.strip()

# Function to load images and process them one by one
def load_images_with_progress(folder_path):
    global annotations, current_index
    try:
        # Clear previous annotations
        annotations.clear()
        current_index = 0
        # Get list of images
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)
        # Process images one by one
        for i, image_file in enumerate(tqdm(image_files, desc="Processing Images")):
            image_path = os.path.join(folder_path, image_file)
            # Save the original image
            original_filename = os.path.basename(image_path)
            original_path = os.path.join(original_dir, original_filename)
            original_image = Image.open(image_path)
            original_image.save(original_path)
            # Detect objects using YOLO
            detections = detect_objects(image_path)
            if detections:
                # Crop the image using YOLO bounding boxes
                cropped_images = crop_image(image_path, detections)
                # Perform OCR on each cropped image
                for j, cropped_image in enumerate(cropped_images):
                    annotation = {
                        "image_path": image_path,
                        "cropped_image": cropped_image,
                        "meter_value": perform_ocr(cropped_image),
                        "original_image": original_image,
                        "room_number": ""  # Initialize room number as empty
                    }
                    annotations.append(annotation)
            else:
                # If no detections, add the original image with no OCR result
                annotation = {
                    "image_path": image_path,
                    "cropped_image": None,
                    "meter_value": "",
                    "original_image": original_image,
                    "room_number": ""  # Initialize room number as empty
                }
                annotations.append(annotation)
            # Update progress bar
            progress_bar.progress((i + 1) / total_images)
        # Update GUI
        if annotations:
            current_index = 0
            update_gui()
    except Exception as e:
        st.error(str(e))

# Function to start loading images in a separate thread
def start_loading_images():
    folder_path = st.session_state.folder_path
    if folder_path:
        threading.Thread(target=load_images_with_progress, args=(folder_path,), daemon=True).start()

# GUI Functions
def update_gui():
    global current_index
    if not annotations:
        return
    annotation = annotations[current_index]
    # Show the original image
    original_image = annotation["original_image"]
    st.image(original_image, caption="Original Image", width=400)
    # Show the cropped image
    cropped_image = annotation["cropped_image"]
    if cropped_image:
        st.image(cropped_image, caption="Cropped Image", width=400)
    # Update meter value text box
    st.text_input("Meter Value", value=annotation.get("meter_value", ""), key="meter_value")
    # Update room number text box
    st.text_input("Room Number", value=annotation.get("room_number", ""), key="room_number")

def next_image():
    global current_index
    if current_index < len(annotations) - 1:
        # Save room number and meter value before moving to the next image
        annotations[current_index]["room_number"] = st.session_state.room_number
        annotations[current_index]["meter_value"] = st.session_state.meter_value
        current_index += 1
        update_gui()

def prev_image():
    global current_index
    if current_index > 0:
        # Save room number and meter value before moving to the previous image
        annotations[current_index]["room_number"] = st.session_state.room_number
        annotations[current_index]["meter_value"] = st.session_state.meter_value
        current_index -= 1
        update_gui()

def export_to_excel():
    from datetime import datetime  # Import for handling dates
    # Prepare data for export
    data = []
    for annotation in annotations:
        data.append({
            "room_number": annotation["room_number"],
            "meter_value": annotation["meter_value"]
        })
    # Create a DataFrame and export to Excel
    df = pd.DataFrame(data)
    output_excel = "results.xlsx"
    df.to_excel(output_excel, index=False)
    # Create a folder with today's date
    today_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    folder_name = f"electricity_meter_values_{today_date}"
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
    # Save original images renamed to room numbers
    for annotation in annotations:
        room_number = annotation["room_number"]
        if room_number:  # Only proceed if room number is not empty
            original_image_path = annotation["image_path"]
            original_image = Image.open(original_image_path)
            # Construct the new filename using the room number
            new_image_name = f"{room_number}.jpg"  # Use room number as the filename
            new_image_path = os.path.join(folder_name, new_image_name)
            # Save the image in the new folder
            original_image.save(new_image_path)
    # Notify the user that the operation is complete
    st.success(f"Excel file and images saved successfully in '{folder_name}' folder.")

# Streamlit App Setup
st.title("Image Cropping Tool")

# Folder Selection Button
st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="folder_path")

# Progress Bar
progress_bar = st.progress(0.0)

# Navigation Buttons
col1, col2 = st.columns(2)
with col1:
    st.button("<< Previous", on_click=prev_image)
with col2:
    st.button("Next >>", on_click=next_image)

# Export Button
st.button("Export to Excel", on_click=export_to_excel)

# Update GUI function with focus handling
update_gui()
