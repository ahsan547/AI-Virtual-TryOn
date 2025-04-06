import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import json

def extract_frame_attributes(frame_path):
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if exists

    # Handle 3-channel images (RGB without alpha)
    if frame.shape[2] == 3:
        # Add a dummy alpha channel (fully opaque)
        alpha_channel = np.ones((frame.shape[0], frame.shape[1]), dtype=frame.dtype) * 255
        frame = cv2.merge((frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], alpha_channel))

    h, w, _ = frame.shape

    # 1. Detect Frame Shape
    gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)  # Use only RGB channels
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape = "unknown"
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        num_sides = len(approx)

        if num_sides >= 5:
            shape = "round"
        elif 4 <= num_sides < 5:
            shape = "rectangular"
        else:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = w_rect / float(h_rect)
            if aspect_ratio > 1.5:
                shape = "cat-eye"
            else:
                shape = "angular"

    # 2. Detect Frame Width
    frame_width = "medium"
    if w < 100:  # Adjust based on your frame sizes
        frame_width = "narrow"
    elif w > 200:
        frame_width = "wide"

    # 3. Extract Dominant Color and Category
    alpha_channel = frame[:, :, 3]
    colored_pixels = frame[:, :, :3][alpha_channel > 0]  # Use RGB channels where alpha > 0
    if len(colored_pixels) == 0:
        colored_pixels = frame[:, :, :3].reshape(-1, 3)  # Fallback to all pixels

    kmeans = KMeans(n_clusters=1)
    kmeans.fit(colored_pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)

    # Classify color category
    red, green, blue = dominant_color
    if red > 200 and green > 150 and blue < 100:
        color_category = "warm"
    elif red < 100 and green < 100 and blue > 150:
        color_category = "cool"
    else:
        color_category = "neutral"

    # 4. Detect Frame Style (simplified)
    style = "classic"
    if shape == "cat-eye" or color_category == "bold":
        style = "bold"
    elif shape == "rectangular" or color_category == "modern":
        style = "modern"

    return {
        "shape": shape,
        "color_category": color_category,
        "frame_width": frame_width,
        "style": style
    }

glasses_dir = "virtualtryon/static/virtualtryon/glasses/"
glasses_data = {}

for frame_file in os.listdir(glasses_dir):
    if frame_file.endswith(".png"):
        frame_path = os.path.join(glasses_dir, frame_file)
        try:
            glasses_data[frame_file] = extract_frame_attributes(frame_path)
            print(f"Processed: {frame_file}")
        except Exception as e:
            print(f"Error processing {frame_file}: {str(e)}")

with open("glasses_attributes.json", "w") as f:
    json.dump(glasses_data, f, indent=4)

print("✅ glasses_attributes.json updated!")