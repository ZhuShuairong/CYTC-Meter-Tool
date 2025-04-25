import os
import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import easyocr
import numpy as np
from datetime import datetime

# Initialize session state variables
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

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
        st.warning(f"Invalid image: {image_path}")
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

# Main Streamlit App
st.title("Electricity Meter Annotation Tool")

# Step 1: Upload Images
st.header("Step 1: Upload Images")
uploaded_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.annotations.clear()
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        # Save the uploaded file temporarily
        image_path = os.path.join(original_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Detect objects using YOLO
        detections = detect_objects(image_path)
        if detections:
            cropped_images = crop_image(image_path, detections)
            for j, cropped_image in enumerate(cropped_images):
                annotation = {
                    "image_path": image_path,
                    "cropped_image": cropped_image,
                    "meter_value": perform_ocr(cropped_image),
                    "room_number": ""  # Initialize room number as empty
                }
                st.session_state.annotations.append(annotation)
        else:
            annotation = {
                "image_path": image_path,
                "cropped_image": None,
                "meter_value": "",
                "room_number": ""  # Initialize room number as empty
            }
            st.session_state.annotations.append(annotation)
        # Update progress bar
        progress_bar.progress((i + 1) / len(uploaded_files))
    st.success("All images processed successfully!")

# Step 2: Annotate Images
if st.session_state.annotations:
    st.header("Step 2: Annotate Images")
    current_index = st.number_input(
        "Image Index",
        min_value=0,
        max_value=len(st.session_state.annotations) - 1,
        value=st.session_state.current_index,
        step=1
    )
    st.session_state.current_index = current_index  # Persist index in session state
    annotation = st.session_state.annotations[current_index]

    # Display original image
    original_image = Image.open(annotation["image_path"])
    st.image(original_image, caption="Original Image", use_column_width=True)

    # Display cropped image
    cropped_image = annotation["cropped_image"]
    if cropped_image:
        st.image(cropped_image, caption="Cropped Image", use_column_width=True)

    # Room Number Input
    room_number = st.text_input("Room Number", value=annotation["room_number"])
    st.session_state.annotations[current_index]["room_number"] = room_number

    # Meter Value Input
    meter_value = st.text_input("Meter Value", value=annotation["meter_value"])
    st.session_state.annotations[current_index]["meter_value"] = meter_value

    # Navigation Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous", disabled=(current_index == 0)):
            st.session_state.current_index -= 1
    with col2:
        if st.button("Next", disabled=(current_index == len(st.session_state.annotations) - 1)):
            st.session_state.current_index += 1

# Step 3: Export Results
if st.session_state.annotations:
    st.header("Step 3: Export Results")
    if st.button("Export to Excel"):
        # Prepare data for export
        data = []
        for annotation in st.session_state.annotations:
            data.append({
                "room_number": annotation["room_number"],
                "meter_value": annotation["meter_value"]
            })
        # Create a DataFrame and export to Excel
        df = pd.DataFrame(data)
        today_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
        folder_name = f"electricity_meter_values_{today_date}"
        os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
        output_path = os.path.join(folder_name, output_excel)
        df.to_excel(output_path, index=False)
        # Save original images renamed to room numbers
        for annotation in st.session_state.annotations:
            room_number = annotation["room_number"]
            if room_number:  # Only proceed if room number is not empty
                original_image_path = annotation["image_path"]
                original_image = Image.open(original_image_path)
                new_image_name = f"{room_number}.jpg"  # Use room number as the filename
                new_image_path = os.path.join(folder_name, new_image_name)
                original_image.save(new_image_path)
        st.success(f"Excel file and images saved successfully in '{folder_name}' folder.")
else:
    st.warning("No images to annotate or export. Please upload images first.")
