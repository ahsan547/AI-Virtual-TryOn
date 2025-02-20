from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from django.conf import settings
import threading
from django.views.decorators.gzip import gzip_page
import time
import logging

logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Debug prints for directory checking
project_dir = settings.BASE_DIR
print(f"Project directory: {project_dir}")

# Get glasses from static folder
glasses_dir = os.path.join(settings.BASE_DIR, 'virtualtryon', 'static', 'virtualtryon', 'glasses')
print(f"Looking for glasses in: {glasses_dir}")

glasses_files = []
if os.path.exists(glasses_dir):
    glasses_files = [f for f in os.listdir(glasses_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    glasses_files.sort()
    print(f"Found glasses files: {glasses_files}")

glasses_list = [os.path.join(glasses_dir, f) for f in glasses_files]
current_glasses_index = 0

# Load precomputed glasses attributes
with open("glasses_attributes.json", "r") as f:
    glasses_data = json.load(f)

# Add these global variables
camera = None
output_frame = None
lock = threading.Lock()

class FaceAnalyzer:
    def __init__(self):
        self.face_shapes = ['oval', 'round', 'square', 'heart', 'rectangular']
        
    def calculate_face_shape(self, landmarks, image_width, image_height):
        try:
            # Convert landmarks to numpy array of (x,y) coordinates
            points = []
            for landmark in landmarks.landmark:
                points.append([landmark.x * image_width, landmark.y * image_height])
            points = np.array(points)
            
            # Calculate face measurements
            # Jawline width (distance between ear points)
            jaw_width = np.linalg.norm(points[234] - points[454])
            
            # Face length (forehead to chin)
            face_length = np.linalg.norm(points[10] - points[152])
            
            # Cheekbone width
            cheekbone_width = np.linalg.norm(points[123] - points[352])
            
            # Determine face shape based on proportions
            ratio_width_length = jaw_width / face_length
            ratio_cheekbone_jaw = cheekbone_width / jaw_width
            
            if ratio_width_length < 0.75:
                return 'rectangular'
            elif ratio_width_length > 0.85 and ratio_cheekbone_jaw < 1.1:
                return 'round'
            elif ratio_width_length > 0.85 and ratio_cheekbone_jaw >= 1.1:
                return 'heart'
            elif ratio_width_length >= 0.75 and ratio_width_length <= 0.85:
                return 'oval'
            else:
                return 'square'
                
        except Exception as e:
            logger.error(f"Error calculating face shape: {str(e)}")
            return 'unknown'
            
    def analyze_skin_tone(self, image, landmarks):
        try:
            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract skin tone from forehead area
            forehead_landmark = landmarks.landmark[151]  # Center of forehead
            x, y = int(forehead_landmark.x * image.shape[1]), int(forehead_landmark.y * image.shape[0])
            
            # Sample area around forehead point
            sample_size = 20
            x1, y1 = max(0, x - sample_size), max(0, y - sample_size)
            x2, y2 = min(image.shape[1], x + sample_size), min(image.shape[0], y + sample_size)
            
            skin_sample = image_rgb[y1:y2, x1:x2]
            average_color = np.mean(skin_sample, axis=(0,1))
            
            # Classify skin tone (simplified version)
            r, g, b = average_color
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            
            if luminance > 200:
                return 'fair'
            elif luminance > 170:
                return 'light'
            elif luminance > 140:
                return 'medium'
            elif luminance > 100:
                return 'olive'
            else:
                return 'dark'
                
        except Exception as e:
            logger.error(f"Error analyzing skin tone: {str(e)}")
            return 'unknown'
    
    def get_glasses_recommendations(self, face_shape, skin_tone):
        recommendations = {
            'round': {
                'recommended': ['glasses2', 'glasses4', 'glasses7', 'glasses9'],  # Angular frames
                'reason': 'Angular and rectangular frames help balance round face features'
            },
            'square': {
                'recommended': ['glasses1', 'glasses3', 'glasses5', 'glasses8'],  # Rounded frames
                'reason': 'Rounded and oval frames soften angular features'
            },
            'oval': {
                'recommended': ['glasses1', 'glasses2', 'glasses3', 'glasses4', 'glasses5'],  # Most frames
                'reason': 'Most frame styles complement oval face shapes well'
            },
            'heart': {
                'recommended': ['glasses3', 'glasses6', 'glasses8', 'glasses9'],  # Bottom-heavy frames
                'reason': 'Bottom-heavy and oval frames balance heart-shaped faces'
            },
            'rectangular': {
                'recommended': ['glasses1', 'glasses5', 'glasses7', 'glasses8'],  # Rounded frames
                'reason': 'Curved and rounded frames help soften angular features'
            }
        }

        # Get base recommendations from face shape
        base_rec = recommendations.get(face_shape, {
            'recommended': ['glasses1', 'glasses2', 'glasses3'],
            'reason': 'Classic frame styles that suit most face shapes'
        })

        # Adjust color recommendations based on skin tone
        color_advice = {
            'fair': 'Consider darker frames for contrast, try glasses4 or glasses7',
            'light': 'Most frame colors work well, especially glasses2, glasses5, and glasses8',
            'medium': 'Both light and dark frames complement your tone, try glasses3 or glasses6',
            'olive': 'Gold or brown tones enhance your complexion, consider glasses5 or glasses9',
            'dark': 'Bold colored frames create striking contrast, try glasses1 or glasses7'
        }

        return {
            'frames': base_rec['recommended'],
            'shape_advice': base_rec['reason'],
            'color_advice': color_advice.get(skin_tone, 'Various frame colors can work for you')
        }

# Create a global instance of FaceAnalyzer
face_analyzer = FaceAnalyzer()

def apply_glasses(image, landmarks, glasses_path):
    try:
        # Read the glasses image
        glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        if glasses is None:
            return None

        # Get facial landmarks for glasses placement
        image_height, image_width = image.shape[:2]
        
        # Get coordinates for eyes
        left_eye = np.mean([[landmarks[33].x * image_width, landmarks[33].y * image_height],
                           [landmarks[133].x * image_width, landmarks[133].y * image_height]], axis=0)
        right_eye = np.mean([[landmarks[362].x * image_width, landmarks[362].y * image_height],
                            [landmarks[263].x * image_width, landmarks[263].y * image_height]], axis=0)

        # Calculate angle for rotation
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate distance between eyes for scaling
        eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        
        # Scale glasses based on eye distance
        glasses_width = glasses.shape[1]
        scale_factor = eye_distance / (glasses_width * 0.4)  # Adjust 0.4 to change glasses size
        
        # Resize glasses
        new_width = int(glasses.shape[1] * scale_factor)
        new_height = int(glasses.shape[0] * scale_factor)
        glasses = cv2.resize(glasses, (new_width, new_height))

        # Rotate glasses
        center = (glasses.shape[1] // 2, glasses.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        glasses = cv2.warpAffine(glasses, rotation_matrix, (glasses.shape[1], glasses.shape[0]))

        # Calculate position to place glasses
        glasses_center_x = int(left_eye[0] + (right_eye[0] - left_eye[0]) / 2)
        glasses_center_y = int(left_eye[1] + (right_eye[1] - left_eye[1]) / 2)
        
        # Calculate top-left corner for glasses placement
        x_offset = int(glasses_center_x - glasses.shape[1] / 2)
        y_offset = int(glasses_center_y - glasses.shape[0] / 2)

        # Ensure offsets are within image bounds
        if x_offset < 0: x_offset = 0
        if y_offset < 0: y_offset = 0

        # Create a region of interest (ROI)
        roi = image[y_offset:min(y_offset + glasses.shape[0], image.shape[0]),
                   x_offset:min(x_offset + glasses.shape[1], image.shape[1])]

        # Create a mask for the glasses
        if glasses.shape[2] == 4:  # If glasses image has an alpha channel
            glasses_alpha = glasses[:roi.shape[0], :roi.shape[1], 3] / 255.0
            glasses_rgb = glasses[:roi.shape[0], :roi.shape[1], :3]

            # Blend the glasses with the ROI
            for c in range(3):  # For each color channel
                roi[:, :, c] = roi[:, :, c] * (1 - glasses_alpha) + glasses_rgb[:, :, c] * glasses_alpha

        return image

    except Exception as e:
        print(f"Error in apply_glasses: {str(e)}")
        return None

def calculate_face_shape(landmarks, width, height):
    # Convert landmarks to numpy array of coordinates
    points = np.array([[landmark.x * width, landmark.y * height] for landmark in landmarks])
    
    # Get face measurements
    face_width = np.linalg.norm(points[234] - points[454])  # Distance between ears
    face_height = np.linalg.norm(points[10] - points[152])  # Distance from chin to forehead
    jaw_width = np.linalg.norm(points[132] - points[361])   # Jawline width
    cheek_width = np.linalg.norm(points[123] - points[352]) # Width at cheekbones
    
    # Calculate ratios
    width_height_ratio = face_width / face_height
    jaw_cheek_ratio = jaw_width / cheek_width
    
    # Determine face shape based on ratios
    if width_height_ratio > 0.85:
        if jaw_cheek_ratio > 0.9:
            shape = "Square"
        else:
            shape = "Round"
    else:
        if jaw_cheek_ratio > 0.9:
            shape = "Rectangle"
        else:
            shape = "Oval"
    
    return {
        'shape': shape,
        'measurements': {
            'width_height_ratio': float(width_height_ratio),
            'jaw_cheek_ratio': float(jaw_cheek_ratio)
        }
    }

def analyze_skin_tone(image, landmarks):
    # Convert landmarks to pixel coordinates
    height, width = image.shape[:2]
    cheek_points = [
        landmarks[123],  # Left cheek
        landmarks[352]   # Right cheek
    ]
    
    # Sample skin tone from cheek areas
    samples = []
    for point in cheek_points:
        x, y = int(point.x * width), int(point.y * height)
        # Sample a small region around the point
        region = image[max(0, y-5):min(height, y+5), max(0, x-5):min(width, x+5)]
        if region.size > 0:
            samples.append(np.mean(region, axis=(0, 1)))
    
    if samples:
        # Average the samples
        avg_color = np.mean(samples, axis=0)
        # Convert BGR to HSV for better color analysis
        hsv_color = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        # Determine skin tone based on value and saturation
        if hsv_color[2] > 200:  # Value
            tone = "Fair"
        elif hsv_color[2] > 150:
            tone = "Medium"
        else:
            tone = "Dark"
    else:
        tone = "Medium"  # Default if sampling fails
    
    return {
        'tone': tone,
        'rgb': [int(c) for c in avg_color]
    }

def get_glasses_recommendations(face_shape, skin_tone):
    recommendations = {
        'Square': {
            'shapes': ['Round', 'Oval', 'Aviator'],
            'colors': {
                'Fair': ['Gold', 'Brown', 'Blue'],
                'Medium': ['Silver', 'Brown', 'Green'],
                'Dark': ['Gold', 'Silver', 'Brown']
            }
        },
        'Round': {
            'shapes': ['Rectangle', 'Square', 'Wayframe'],
            'colors': {
                'Fair': ['Black', 'Silver', 'Blue'],
                'Medium': ['Brown', 'Gold', 'Green'],
                'Dark': ['Silver', 'Gold', 'Black']
            }
        },
        'Oval': {
            'shapes': ['Any Shape Works Well', 'Square', 'Rectangle'],
            'colors': {
                'Fair': ['Brown', 'Black', 'Silver'],
                'Medium': ['Gold', 'Brown', 'Green'],
                'Dark': ['Gold', 'Silver', 'Black']
            }
        },
        'Rectangle': {
            'shapes': ['Round', 'Oval', 'Butterfly'],
            'colors': {
                'Fair': ['Gold', 'Brown', 'Blue'],
                'Medium': ['Silver', 'Black', 'Brown'],
                'Dark': ['Gold', 'Silver', 'Black']
            }
        }
    }
    
    return {
        'recommended_shapes': recommendations[face_shape]['shapes'],
        'recommended_colors': recommendations[face_shape]['colors'][skin_tone]
    }

def score_glasses(user_attrs, frame_attrs):
    score = 0
    
    # 1. Face Shape Compatibility (40% weight)
    shape_compatibility = {
        "round": {
            "rectangular": 40,
            "square": 35,
            "angular": 35,
            "cat-eye": 30,
            "oval": 20,
            "round": 15
        },
        "square": {
            "round": 40,
            "oval": 35,
            "cat-eye": 30,
            "rectangular": 20,
            "angular": 15
        },
        "oval": {
            "geometric": 40,
            "cat-eye": 35,
            "rectangular": 30,
            "round": 30,
            "square": 25
        },
        "heart": {
            "oval": 40,
            "round": 35,
            "cat-eye": 30,
            "rectangular": 25
        },
        "diamond": {
            "cat-eye": 40,
            "oval": 35,
            "round": 30,
            "rectangular": 25
        },
        "rectangular": {
            "round": 40,
            "oval": 35,
            "cat-eye": 30,
            "square": 20
        }
    }
    
    score += shape_compatibility.get(user_attrs["face_shape"], {}).get(frame_attrs["shape"], 20)
    
    # 2. Color Harmony (40% weight)
    color_harmony = {
        ("warm", "light"): {
            "gold": 40,
            "tortoise": 35,
            "brown": 30,
            "silver": 20
        },
        ("warm", "medium"): {
            "tortoise": 40,
            "gold": 35,
            "brown": 35,
            "silver": 25
        },
        ("warm", "dark"): {
            "gold": 40,
            "brown": 35,
            "tortoise": 30,
            "silver": 25
        },
        ("cool", "light"): {
            "silver": 40,
            "black": 35,
            "blue": 30,
            "gold": 20
        },
        ("cool", "medium"): {
            "silver": 40,
            "blue": 35,
            "black": 35,
            "gold": 25
        },
        ("cool", "dark"): {
            "silver": 40,
            "black": 35,
            "blue": 30,
            "gold": 25
        }
    }
    
    tone_key = (user_attrs["skin_tone"], user_attrs["skin_brightness"])
    score += color_harmony.get(tone_key, {}).get(frame_attrs["color_category"], 20)
    
    # 3. Size Proportion (20% weight)
    size_compatibility = {
        "small": {"narrow": 20, "medium": 15, "wide": 5},
        "medium": {"medium": 20, "narrow": 15, "wide": 15},
        "large": {"wide": 20, "medium": 15, "narrow": 5}
    }
    
    score += size_compatibility.get(user_attrs["face_size"], {}).get(frame_attrs["frame_width"], 10)
    
    return score

def get_top_glasses(user_attrs):
    scores = []
    
    # Print user attributes for debugging
    print(f"User Attributes: {user_attrs}")
    
    for i, (frame_file, frame_attrs) in enumerate(glasses_data.items()):
        score = score_glasses(user_attrs, frame_attrs)
        scores.append((frame_file, score, i))  # Include index to ensure uniqueness
        print(f"Frame: {frame_file}, Score: {score}, Index: {i}")
    
    # Sort by score (descending) and then by index (ascending)
    scores.sort(key=lambda x: (-x[1], x[2]))
    
    # Get top 3 unique frames
    top_frames = []
    seen_scores = set()
    
    for frame, score, idx in scores:
        if score not in seen_scores:
            top_frames.append(frame)
            seen_scores.add(score)
            if len(top_frames) == 3:
                break
    
    # If we don't have 3 unique scores, fill remaining slots with next best frames
    while len(top_frames) < 3 and len(top_frames) < len(scores):
        frame = scores[len(top_frames)][0]
        if frame not in top_frames:
            top_frames.append(frame)
    
    print(f"Selected top frames: {top_frames}")  # Debug print
    return top_frames

def process_image(image_path, glasses_index):
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        frame = apply_glasses(frame, results.multi_face_landmarks[0].landmark, glasses_list[glasses_index])

    return frame

@gzip_page
def video_feed(request):
    global current_glasses_index
    if request.GET.get('glasses_index'):
        current_glasses_index = int(request.GET.get('glasses_index'))
    return StreamingHttpResponse(generate_frames(),
                               content_type='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global current_glasses_index
    camera = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Process the frame with face mesh
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                # Apply glasses if face is detected
                landmarks = results.multi_face_landmarks[0]
                frame = apply_glasses(frame, landmarks.landmark, glasses_list[current_glasses_index])
            
            # Convert frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

def stop_camera(request):
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
            print("Camera stopped successfully")
        return JsonResponse({'status': 'success'})
    except Exception as e:
        print(f"Error stopping camera: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)})

def reset_camera(request):
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

def upload_image(request):
    if request.method == 'POST':
        if 'glasses_index' in request.POST:
            # Handle glasses switching
            glasses_index = int(request.POST['glasses_index'])
            if 'last_uploaded_image' in request.session:
                image_path = request.session['last_uploaded_image']
                image = cv2.imread(image_path)
                if image is not None:
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]
                        processed_image = apply_glasses(image, landmarks.landmark, glasses_list[glasses_index])
                        if processed_image is not None:
                            fs = FileSystemStorage()
                            filename = os.path.basename(image_path)
                            output_path = fs.path('processed_' + filename)
                            cv2.imwrite(output_path, processed_image)
                            return JsonResponse({'processed_url': fs.url('processed_' + filename)})
            return JsonResponse({'error': 'Failed to process image'})
        
        elif 'image' in request.FILES:
            # Handle new image upload
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            request.session['last_uploaded_image'] = file_path
            
            image = cv2.imread(file_path)
            if image is not None:
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    
                    # Get face analysis
                    face_shape = face_analyzer.calculate_face_shape(landmarks, image.shape[1], image.shape[0])
                    skin_tone = face_analyzer.analyze_skin_tone(image, landmarks)
                    recommendations = face_analyzer.get_glasses_recommendations(face_shape, skin_tone)
                    
                    processed_image = apply_glasses(image, landmarks.landmark, glasses_list[0])
                    if processed_image is not None:
                        output_path = fs.path('processed_' + filename)
                        cv2.imwrite(output_path, processed_image)
                        return JsonResponse({
                            'file_url': fs.url(filename),
                            'processed_url': fs.url('processed_' + filename),
                            'face_shape': face_shape,
                            'skin_tone': skin_tone,
                            'recommendations': recommendations
                        })
    
    return JsonResponse({'error': 'Invalid request'})

def index(request):
    return render(request, 'virtualtryon/home.html')

def try_on(request):
    return render(request, 'virtualtryon/try_on.html', {
        'glasses_files': glasses_files,
        'glasses_list': glasses_list,
        'glasses_range': range(len(glasses_files))
    })

def contact(request):
    if request.method == 'POST':
        # Handle form submission here
        # You can add email functionality later
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        # Add your email sending logic here
        
        # For now, just redirect back to contact page
        return redirect('contact')
    return render(request, 'virtualtryon/contact.html')

def about(request):
    return render(request, 'virtualtryon/about.html')

def webcam_try_on(request):
    return render(request, 'virtualtryon/webcam.html', {
        'glasses_files': glasses_files,
        'glasses_list': glasses_list,
        'glasses_range': range(len(glasses_files))
    })