import os
import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import easyocr
import numpy as np
from datetime import datetime
import shutil

# Initialize session state variables
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "model_initialized" not in st.session_state:
    st.session_state.model_initialized = False

# Directories for saving original and YOLO-ed images
original_dir = "original_images"
yoloed_dir = "yoloed_images"

# Ensure directories exist
os.makedirs(original_dir, exist_ok=True)
os.makedirs(yoloed_dir, exist_ok=True)

# Cache the YOLO model initialization
@st.cache_resource
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}. Ensure the model is in the correct location.")
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.stop()

# Cache the EasyOCR reader initialization
@st.cache_resource
def initialize_reader():
    return easyocr.Reader(['en'])

# Initialize YOLO model and EasyOCR reader only once
if not st.session_state.get("model_initialized", False):
    model = load_model("best.pt")
    reader = initialize_reader()
    st.session_state["model"] = model
    st.session_state["reader"] = reader
    st.session_state["model_initialized"] = True
else:
    model = st.session_state["model"]
    reader = st.session_state["reader"]

# Function to run YOLO inference on a single image
@st.cache_data
def detect_objects(image_path):
    if not is_valid_image(image_path):
        st.warning(f"Invalid image: {image_path}")
        return []
    
    # Validate the model before use
    if model is None:
        st.error("Model is not initialized. Please check the model file and try again.")
        st.stop()
    
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
        global_crop_counter += 1
        cropped_filename = f"crop_{global_crop_counter}.jpg"
        cropped_path = os.path.join(yoloed_dir, cropped_filename)
        cropped_image.save(cropped_path)
        cropped_images.append(cropped_image)
    return cropped_images

# Function to perform OCR on a cropped image
@st.cache_data
def perform_ocr(image):
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0)
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
        image_path = os.path.join(original_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Check if results are cached
        cached_annotation = next((ann for ann in st.session_state.annotations if ann["image_path"] == image_path), None)
        if cached_annotation:
            detections = cached_annotation.get("detections", [])
            ocr_results = cached_annotation.get("ocr_results", [""])
        else:
            # Perform YOLO detection and OCR only once during upload
            detections = detect_objects(image_path)
            cropped_images = crop_image(image_path, detections) if detections else []
            ocr_results = [perform_ocr(cropped_image) for cropped_image in cropped_images] if cropped_images else [""]
        
        # Store annotations in session state
        for j, cropped_image in enumerate(cropped_images or []):
            annotation = {
                "image_path": image_path,
                "cropped_image": cropped_image,
                "meter_value": ocr_results[j],
                "room_number": "",
                "detections": detections,  # Cache detections
                "ocr_results": ocr_results  # Cache OCR results
            }
            st.session_state.annotations.append(annotation)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    st.success("All images processed successfully!")

# Step 2: Annotate Images
if st.session_state.annotations:
    st.header("Step 2: Annotate Images")
    # Constrain current_index to valid bounds
    max_index = len(st.session_state.annotations) - 1
    current_index = st.number_input(
        "Image Index",
        min_value=0,
        max_value=max_index,
        value=st.session_state.current_index,
        step=1
    )
    st.session_state.current_index = current_index
    annotation = st.session_state.annotations[current_index]

    # Display original image
    original_image = Image.open(annotation["image_path"])
    st.image(original_image, caption="Original Image", use_container_width=True)

    # Display cropped image
    cropped_image = annotation["cropped_image"]
    if cropped_image:
        st.image(cropped_image, caption="Cropped Image", use_container_width=True)

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
            # Save current annotations before moving
            st.session_state.annotations[current_index]["room_number"] = room_number
            st.session_state.annotations[current_index]["meter_value"] = meter_value
            st.session_state.current_index -= 1
            # Reset room number for new image
            st.session_state.annotations[st.session_state.current_index]["room_number"] = ""
    with col2:
        if st.button("Next", disabled=(current_index == max_index)):
            # Save current annotations before moving
            st.session_state.annotations[current_index]["room_number"] = room_number
            st.session_state.annotations[current_index]["meter_value"] = meter_value
            st.session_state.current_index += 1
            # Reset room number for new image
            st.session_state.annotations[st.session_state.current_index]["room_number"] = ""

# Step 3: Export Results
if st.session_state.annotations:
    st.header("Step 3: Export Results")
    if st.button("Export to Excel"):
        output_excel = "results.xlsx"
        folder_name = f"electricity_meter_values_{datetime.now().strftime('%Y-%m-%d')}"
        os.makedirs(folder_name, exist_ok=True)
        output_path = os.path.join(folder_name, output_excel)

        # Prepare data for export
        data = [{"room_number": ann["room_number"], "meter_value": ann["meter_value"]} for ann in st.session_state.annotations]
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)

        # Save images renamed to room numbers
        for annotation in st.session_state.annotations:
            room_number = annotation["room_number"]
            if room_number:
                original_image_path = annotation["image_path"]
                original_image = Image.open(original_image_path)
                new_image_name = f"{room_number}.jpg"
                new_image_path = os.path.join(folder_name, new_image_name)
                original_image.save(new_image_path)

        # Create a zip archive of the folder
        shutil.make_archive(folder_name, 'zip', folder_name)
        st.success(f"Zip file '{folder_name}.zip' created successfully.")
else:
    st.warning("No images to annotate or export. Please upload images first.")
