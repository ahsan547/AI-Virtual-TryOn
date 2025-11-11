"""
ML Service for Virtual Try-On Recommendations
Loads trained models and provides recommendation services
"""

import os
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from django.conf import settings

# Load models and data
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"

# Load model info
try:
    with open(MODELS_DIR / "face_shape_model_info.json", 'r') as f:
        FACE_SHAPE_INFO = json.load(f)
    FACE_SHAPE_CLASSES = FACE_SHAPE_INFO['classes']
except FileNotFoundError:
    FACE_SHAPE_CLASSES = ['oval', 'round', 'square', 'heart', 'rectangular', 'oblong']
    FACE_SHAPE_INFO = {'classes': FACE_SHAPE_CLASSES}

try:
    with open(MODELS_DIR / "skin_tone_model_info.json", 'r') as f:
        SKIN_TONE_INFO = json.load(f)
    SKIN_TONE_CLASSES = SKIN_TONE_INFO['classes']
except FileNotFoundError:
    SKIN_TONE_CLASSES = ['fair', 'light', 'medium', 'olive', 'dark', 'black', 'brown', 'white']
    SKIN_TONE_INFO = {'classes': SKIN_TONE_CLASSES}

# Load recommendation model
try:
    with open(MODELS_DIR / "recommendation_model.pkl", 'rb') as f:
        RECOMMENDATION_MODEL = pickle.load(f)
    with open(MODELS_DIR / "recommendation_encoders.pkl", 'rb') as f:
        RECOMMENDATION_ENCODERS = pickle.load(f)
except FileNotFoundError:
    RECOMMENDATION_MODEL = None
    RECOMMENDATION_ENCODERS = None

# Load enhanced recommendations
try:
    with open(MODELS_DIR / "enhanced_recommendations.json", 'r') as f:
        ENHANCED_RECOMMENDATIONS = json.load(f)
except FileNotFoundError:
    ENHANCED_RECOMMENDATIONS = {}


def normalize_face_shape(shape):
    """Normalize face shape to match model classes"""
    shape_lower = shape.lower().strip()
    
    # Map variations to standard shapes
    mapping = {
        'rectangular': 'oblong',
        'long': 'oblong',
        'diamond': 'heart',
        'triangle': 'heart'
    }
    
    return mapping.get(shape_lower, shape_lower)


def normalize_skin_tone(tone):
    """Normalize skin tone to match model classes"""
    tone_lower = tone.lower().strip()
    
    # Map variations to standard tones
    mapping = {
        'fair': 'white',
        'light': 'white',
        'medium': 'brown',
        'olive': 'brown',
        'dark': 'black'
    }
    
    return mapping.get(tone_lower, tone_lower)


def get_ml_recommendations(face_shape, skin_tone):
    """
    Get ML-based recommendations for glasses based on face shape and skin tone
    
    Args:
        face_shape: Detected face shape
        skin_tone: Detected skin tone
    
    Returns:
        dict with recommended frames and advice
    """
    # Normalize inputs
    normalized_shape = normalize_face_shape(face_shape)
    normalized_tone = normalize_skin_tone(skin_tone)
    
    # Try to get enhanced recommendations from dataset
    if normalized_shape in ENHANCED_RECOMMENDATIONS:
        rec_data = ENHANCED_RECOMMENDATIONS[normalized_shape]
        recommended_frames = rec_data.get('recommended', [])
        
        # Map frame shapes to your glasses files
        frame_to_glasses = {
            'round': ['glasses1', 'glasses2', 'glasses4', 'glasses5'],
            'oval': ['glasses1', 'glasses3', 'glasses5', 'glasses8'],
            'rectangle': ['glasses3', 'glasses6', 'glasses7', 'glasses8', 'glasses9'],
            'square': ['glasses2', 'glasses4'],
            'cat eye': ['glasses6', 'glasses9'],
            'geometric': ['glasses7', 'glasses9'],
            'browline': ['glasses8']
        }
        
        # Get glasses recommendations
        glasses_recommendations = []
        for frame_shape in recommended_frames:
            frame_lower = frame_shape.lower()
            if frame_lower in frame_to_glasses:
                glasses_recommendations.extend(frame_to_glasses[frame_lower])
        
        # Remove duplicates and limit to top 3
        glasses_recommendations = list(dict.fromkeys(glasses_recommendations))[:3]
        
        return {
            'frames': glasses_recommendations if glasses_recommendations else ['glasses1', 'glasses2', 'glasses3'],
            'shape_advice': f'Based on your {normalized_shape} face shape, these frame styles are recommended.',
            'color_advice': get_color_advice(normalized_tone),
            'confidence': rec_data.get('confidence', 0.5)
        }
    
    # Fallback to default recommendations
    return get_default_recommendations(normalized_shape, normalized_tone)


def get_color_advice(skin_tone):
    """Get color advice based on skin tone"""
    advice_map = {
        'white': 'Light and medium-toned frames work well. Consider silver, gold, or brown frames.',
        'brown': 'Both light and dark frames complement your tone. Try tortoise, brown, or black frames.',
        'black': 'Bold colored frames create striking contrast. Consider black, gold, or colorful frames.'
    }
    
    return advice_map.get(skin_tone, 'Various frame colors can work for you. Experiment with different styles.')


def get_default_recommendations(face_shape, skin_tone):
    """Default recommendations if ML model data is not available"""
    recommendations = {
        'round': {
            'recommended': ['glasses2', 'glasses4', 'glasses7', 'glasses9'],
            'reason': 'Angular and rectangular frames help balance round face features'
        },
        'square': {
            'recommended': ['glasses1', 'glasses3', 'glasses5', 'glasses8'],
            'reason': 'Rounded and oval frames soften angular features'
        },
        'oval': {
            'recommended': ['glasses1', 'glasses2', 'glasses3', 'glasses4', 'glasses5'],
            'reason': 'Most frame styles complement oval face shapes well'
        },
        'heart': {
            'recommended': ['glasses3', 'glasses6', 'glasses8', 'glasses9'],
            'reason': 'Bottom-heavy and oval frames balance heart-shaped faces'
        },
        'oblong': {
            'recommended': ['glasses1', 'glasses5', 'glasses7', 'glasses8'],
            'reason': 'Curved and rounded frames help soften angular features'
        },
        'rectangular': {
            'recommended': ['glasses1', 'glasses5', 'glasses7', 'glasses8'],
            'reason': 'Curved and rounded frames help soften angular features'
        }
    }
    
    base_rec = recommendations.get(face_shape, {
        'recommended': ['glasses1', 'glasses2', 'glasses3'],
        'reason': 'Classic frame styles that suit most face shapes'
    })
    
    color_advice = get_color_advice(skin_tone)
    
    return {
        'frames': base_rec['recommended'],
        'shape_advice': base_rec['reason'],
        'color_advice': color_advice,
        'confidence': 0.5
    }


def predict_suitability(face_shape, frame_shape):
    """
    Predict if a frame shape is suitable for a face shape using ML model
    
    Args:
        face_shape: Detected face shape
        frame_shape: Frame shape to check
    
    Returns:
        float: Suitability score (0-1)
    """
    if RECOMMENDATION_MODEL is None or RECOMMENDATION_ENCODERS is None:
        return 0.5  # Default neutral score
    
    try:
        # Encode face shape
        face_encoder = RECOMMENDATION_ENCODERS.get('face_shape')
        frame_encoder = RECOMMENDATION_ENCODERS.get('frame_shape')
        
        if face_encoder is None or frame_encoder is None:
            return 0.5
        
        normalized_face = normalize_face_shape(face_shape)
        normalized_frame = frame_shape.lower().strip()
        
        # Try to encode
        try:
            face_encoded = face_encoder.transform([normalized_face])[0]
        except (ValueError, KeyError):
            # If face shape not in training data, return neutral
            return 0.5
        
        try:
            frame_encoded = frame_encoder.transform([normalized_frame])[0]
        except (ValueError, KeyError):
            # If frame shape not in training data, return neutral
            return 0.5
        
        # Predict
        X = np.array([[face_encoded]])
        prediction = RECOMMENDATION_MODEL.predict_proba(X)[0]
        
        # Return probability of being suitable
        return prediction[1] if len(prediction) > 1 else 0.5
    
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return 0.5

