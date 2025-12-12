import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image
import io
import numpy as np
import torch
from deepface import DeepFace
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Multimodal Sentimental Analysis",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        text-align: center;
        color: grey;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        transform: scale(1.02);
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .model-branch {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 3px solid;
    }
    .score-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎭 Multimodal Sentimental Analysis App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ML-based Text & Image Analysis App by Priyanshu Joarder</p>', unsafe_allow_html=True)

# Initialize session state
if 'text_history' not in st.session_state:
    st.session_state.text_history = []
if 'image_analysis_result' not in st.session_state:
    st.session_state.image_analysis_result = None

# ==================== TEXT ANALYSIS MODELS ====================

@st.cache_resource
def load_text_models():
    """Load text sentiment analysis model (unified for both branches)"""
    try:
        # Use the same model for both positive and negative analysis
        unified_model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )
        
        return unified_model
    except Exception as e:
        st.error(f"Error loading text model: {e}")
        return None

@st.cache_resource
def load_image_models():
    """Load image analysis models"""
    try:
        classifier = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            device=-1
        )
        return classifier
    except Exception as e:
        st.warning(f"Could not load image model: {e}")
        return None

# ==================== TEXT ANALYSIS FUNCTIONS ====================

def analyze_with_positive_branch(text, unified_model):
    """Analyze text with positive detection model"""
    if not unified_model or not text.strip():
        return None
    
    try:
        result = unified_model(text[:512])[0]
        label = result['label']
        score = result['score']
        stars = int(label.split()[0])
        positive_confidence = (stars - 1) / 4.0
        positive_confidence = positive_confidence * score
        
        return {
            'confidence': positive_confidence,
            'raw_score': score,
            'stars': stars,
            'interpretation': f"{stars} stars"
        }
    except Exception as e:
        return None

def analyze_with_negative_branch(text, unified_model):
    """Analyze text with negative detection model (using same model)"""
    if not unified_model or not text.strip():
        return None
    
    try:
        result = unified_model(text[:512])[0]
        label = result['label']
        score = result['score']
        stars = int(label.split()[0])
        
        # Invert the stars rating to get negative confidence
        # 1-2 stars = high negative confidence
        # 4-5 stars = low negative confidence
        negative_confidence = (5 - stars) / 4.0
        negative_confidence = negative_confidence * score
        
        return {
            'confidence': negative_confidence,
            'raw_score': score,
            'stars': stars,
            'interpretation': f"{stars} stars (inverted)"
        }
    except Exception as e:
        return None

def detect_content_features(text):
    """Detect special content features"""
    text_lower = text.lower()
    
    positive_words = ['love', 'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'awesome', 
                      'perfect', 'brilliant', 'outstanding', 'superb', 'happy', 'joy', 'delighted']
    negative_words = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'bad', 'poor', 'disappointing',
                      'useless', 'pathetic', 'disgusting', 'annoying']
    violent_words = ['kill', 'murder', 'destroy', 'attack', 'hurt', 'harm', 'death', 'violent', 
                     'threat', 'weapon', 'bomb', 'shoot']
    negation_words = ['not', 'no', 'never', 'nothing', 'neither', 'nobody', "n't", 'dont', "don't"]
    
    features = {
        'positive_count': sum(1 for word in positive_words if word in text_lower),
        'negative_count': sum(1 for word in negative_words if word in text_lower),
        'violent_count': sum(1 for word in violent_words if word in text_lower),
        'negation_count': sum(1 for word in negation_words if word in text_lower),
        'has_violence': any(word in text_lower for word in violent_words),
        'has_strong_positive': any(word in text_lower for word in positive_words[:8]),
        'has_strong_negative': any(word in text_lower for word in negative_words[:8])
    }
    
    return features

def combine_branch_outputs(positive_result, negative_result, strategy, threshold, pos_weight, features, apply_penalties):
    """Combine text analysis outputs"""
    pos_conf = positive_result['confidence']
    neg_conf = negative_result['confidence']
    
    if apply_penalties:
        if features['has_violence'] and pos_conf > neg_conf:
            pos_conf = pos_conf * 0.2
        if features['has_strong_positive'] and features['positive_count'] > features['negative_count']:
            pos_conf = min(1.0, pos_conf * 1.2)
        if features['has_strong_negative'] and features['negative_count'] > features['positive_count']:
            neg_conf = min(1.0, neg_conf * 1.2)
        if features['negation_count'] >= 2:
            if features['has_strong_positive'] and features['negation_count'] > 0:
                pos_conf = pos_conf * 0.6
                neg_conf = min(1.0, neg_conf * 1.3)
    
    if strategy == "Score Difference":
        final_score = pos_conf - neg_conf
        confidence = abs(final_score)
    elif strategy == "Weighted Average":
        neg_weight = 1.0 - pos_weight
        final_score = (pos_conf * pos_weight) - (neg_conf * neg_weight)
        confidence = (pos_conf * pos_weight + neg_conf * neg_weight)
    elif strategy == "Confidence Voting":
        if pos_conf > neg_conf:
            final_score = pos_conf
            confidence = pos_conf
        else:
            final_score = -neg_conf
            confidence = neg_conf
    elif strategy == "Threshold-Based":
        diff = pos_conf - neg_conf
        if abs(diff) < threshold:
            final_score = 0.0
            confidence = 0.5
        else:
            final_score = diff
            confidence = abs(diff)
    else:  # Neural Fusion
        alpha = 2.0
        fusion = (pos_conf - neg_conf) * alpha
        final_score = np.tanh(fusion)
        confidence = (pos_conf + neg_conf) / 2
    
    if abs(final_score) < threshold:
        label = "NEUTRAL"
        sentiment = "Neutral 😐"
        color = "#FFE5B4"
    elif final_score > 0:
        label = "POSITIVE"
        sentiment = "Positive 😊"
        color = "#90EE90"
    else:
        label = "NEGATIVE"
        sentiment = "Negative 😞"
        color = "#FFB6C6"
    
    return {
        'sentiment': sentiment,
        'label': label,
        'final_score': final_score,
        'confidence': confidence,
        'color': color,
        'positive_score': pos_conf,
        'negative_score': neg_conf,
        'score_difference': pos_conf - neg_conf,
        'positive_branch': positive_result,
        'negative_branch': negative_result,
        'features': features
    }

def analyze_text_dual_branch(text, unified_model, strategy, threshold, pos_weight, apply_penalties):
    """Main text analysis function"""
    if not text.strip():
        return None
    
    features = detect_content_features(text)
    positive_result = analyze_with_positive_branch(text, unified_model)
    if not positive_result:
        return None
    
    negative_result = analyze_with_negative_branch(text, unified_model)
    if not negative_result:
        return None
    
    return combine_branch_outputs(positive_result, negative_result, strategy, threshold, pos_weight, features, apply_penalties)

# ==================== IMAGE ANALYSIS FUNCTIONS ====================

def analyze_with_deepface(image):
    """Primary emotion analysis using DeepFace"""
    try:
        img_array = np.array(image)
        backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
        
        for backend in backends:
            try:
                analysis = DeepFace.analyze(
                    img_array, 
                    actions=['emotion', 'age', 'gender', 'race'],
                    enforce_detection=False,
                    detector_backend=backend,
                    silent=True
                )
                
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                emotions = analysis['emotion']
                max_score = max(emotions.values())
                confidence = min(int(max_score * 1.2), 100)
                
                return {
                    "detected": True,
                    "emotions": emotions,
                    "dominant_emotion": analysis['dominant_emotion'],
                    "confidence": confidence,
                    "age": analysis.get('age', 'N/A'),
                    "gender": analysis.get('dominant_gender', 'N/A'),
                    "race": analysis.get('dominant_race', 'N/A'),
                    "backend": backend
                }
            except:
                continue
        
        return {"detected": False, "error": "No face detected"}
    except Exception as e:
        return {"detected": False, "error": str(e)}

def analyze_with_huggingface(image, classifier):
    """Secondary emotion analysis using Hugging Face"""
    try:
        if classifier is None:
            return {"detected": False}
        
        results = classifier(image)
        emotions_dict = {}
        emotion_mapping = {
            'happy': 'happy', 'sad': 'sad', 'angry': 'angry',
            'fear': 'fear', 'surprise': 'surprise', 'neutral': 'neutral', 'disgust': 'disgust'
        }
        
        for result in results:
            label = result['label'].lower()
            score = result['score'] * 100
            for key, value in emotion_mapping.items():
                if key in label or value in label:
                    emotions_dict[value] = score
                    break
        
        if emotions_dict:
            dominant = max(emotions_dict, key=emotions_dict.get)
            return {"detected": True, "emotions": emotions_dict, "dominant_emotion": dominant}
        
        return {"detected": False}
    except Exception as e:
        return {"detected": False, "error": str(e)}

def combine_emotion_analyses(deepface_result, hf_result):
    """Combine image analysis results"""
    if not deepface_result['detected'] and not hf_result['detected']:
        return {"detected": False}
    
    if not deepface_result['detected']:
        return hf_result
    if not hf_result['detected']:
        return deepface_result
    
    combined_emotions = {}
    all_emotions = set(deepface_result['emotions'].keys()) | set(hf_result['emotions'].keys())
    
    for emotion in all_emotions:
        df_score = deepface_result['emotions'].get(emotion, 0)
        hf_score = hf_result['emotions'].get(emotion, 0)
        combined_emotions[emotion] = (df_score * 0.7 + hf_score * 0.3)
    
    dominant = max(combined_emotions, key=combined_emotions.get)
    confidence = min(int(combined_emotions[dominant] * 1.1), 100)
    
    return {
        "detected": True,
        "emotions": combined_emotions,
        "dominant_emotion": dominant,
        "confidence": confidence,
        "age": deepface_result.get('age', 'N/A'),
        "gender": deepface_result.get('gender', 'N/A'),
        "method": "Combined Analysis"
    }

def analyze_colors(image):
    """Enhanced color analysis"""
    img_array = np.array(image)
    small_img = cv2.resize(img_array, (200, 200))
    hsv_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2HSV)
    
    avg_color = np.mean(small_img.reshape(-1, 3), axis=0)
    r, g, b = avg_color
    
    temp = "warm" if r > b else "cool"
    temp_desc = "energetic and stimulating" if r > b else "calming and soothing"
    
    avg_saturation = np.mean(hsv_img[:,:,1])
    brightness_level = int((r + g + b) / 3)
    
    emotional_impacts = []
    if brightness_level > 180:
        emotional_impacts.append("bright and uplifting")
    elif brightness_level < 80:
        emotional_impacts.append("dark and mysterious")
    
    if avg_saturation > 150:
        emotional_impacts.append("vibrant and lively")
    elif avg_saturation < 50:
        emotional_impacts.append("muted and subtle")
    
    if r > g and r > b:
        emotional_impacts.append("passionate and exciting (red dominance)")
    elif b > r and b > g:
        emotional_impacts.append("calm and trustworthy (blue dominance)")
    elif g > r and g > b:
        emotional_impacts.append("natural and balanced (green dominance)")
    
    return {
        "dominant_color": temp,
        "emotional_impact": ", ".join(emotional_impacts) if emotional_impacts else temp_desc,
        "brightness": int(brightness_level),
        "saturation": int(avg_saturation),
        "rgb_values": {"R": int(r), "G": int(g), "B": int(b)},
        "temperature": temp_desc
    }

def analyze_composition(image):
    """Enhanced composition analysis"""
    img_array = np.array(image.convert('L'))
    contrast = img_array.std()
    brightness_mean = img_array.mean()
    
    edges = cv2.Canny(img_array, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size * 100
    
    if contrast > 70:
        contrast_sentiment = "highly dynamic and dramatic"
        contrast_emotion = "intense"
    elif contrast > 40:
        contrast_sentiment = "balanced and engaging"
        contrast_emotion = "moderate"
    else:
        contrast_sentiment = "soft and gentle"
        contrast_emotion = "calm"
    
    if brightness_mean > 180:
        brightness_sentiment = "very bright and cheerful"
        brightness_emotion = "optimistic"
    elif brightness_mean > 130:
        brightness_sentiment = "well-lit and positive"
        brightness_emotion = "pleasant"
    elif brightness_mean > 80:
        brightness_sentiment = "moderately lit"
        brightness_emotion = "neutral"
    else:
        brightness_sentiment = "dark and moody"
        brightness_emotion = "somber"
    
    if edge_density > 15:
        complexity = "complex and detailed"
    elif edge_density > 8:
        complexity = "moderate detail"
    else:
        complexity = "simple and minimalist"
    
    return {
        "contrast": float(contrast),
        "contrast_sentiment": contrast_sentiment,
        "contrast_emotion": contrast_emotion,
        "brightness_mean": float(brightness_mean),
        "brightness_sentiment": brightness_sentiment,
        "brightness_emotion": brightness_emotion,
        "edge_density": float(edge_density),
        "complexity": complexity
    }

def calculate_overall_sentiment(results):
    """Calculate overall image sentiment"""
    sentiment_score = 0
    confidence = 0
    factors = []
    
    if results['facial_emotions']['detected']:
        emotions = results['facial_emotions']['emotions']
        positive = emotions.get('happy', 0) + emotions.get('surprise', 0) * 0.5
        negative = (emotions.get('sad', 0) + emotions.get('angry', 0) + 
                   emotions.get('fear', 0) + emotions.get('disgust', 0) * 0.8)
        neutral = emotions.get('neutral', 0)
        
        emotion_confidence = results['facial_emotions'].get('confidence', 70)
        
        if positive > negative and positive > neutral:
            sentiment_score += 50
            factors.append(f"Positive facial expression ({results['facial_emotions']['dominant_emotion']})")
        elif negative > positive and negative > neutral:
            sentiment_score -= 50
            factors.append(f"Negative facial expression ({results['facial_emotions']['dominant_emotion']})")
        else:
            factors.append("Neutral facial expression")
        
        confidence += emotion_confidence * 0.6
    else:
        confidence += 30
        factors.append("No face detected - analyzing visual elements")
    
    brightness = results['color_analysis']['brightness']
    saturation = results['color_analysis']['saturation']
    
    if brightness > 160:
        sentiment_score += 20
        factors.append("Bright colors suggest positivity")
    elif brightness < 90:
        sentiment_score -= 15
        factors.append("Dark colors suggest seriousness")
    
    if saturation > 150:
        sentiment_score += 10
        factors.append("High saturation indicates vibrancy")
    
    confidence += 25
    
    comp_brightness = results['composition']['brightness_mean']
    if comp_brightness > 150:
        sentiment_score += 10
        factors.append("Well-lit composition")
    elif comp_brightness < 80:
        sentiment_score -= 10
        factors.append("Low-key lighting")
    
    confidence += 15
    
    if sentiment_score > 25:
        final_sentiment = "positive"
        description = "The image conveys positive sentiment"
    elif sentiment_score < -25:
        final_sentiment = "negative"
        description = "The image conveys negative sentiment"
    else:
        final_sentiment = "neutral"
        description = "The image has neutral sentiment"
    
    return {
        "sentiment": final_sentiment,
        "confidence": min(int(confidence), 100),
        "score": sentiment_score,
        "description": description,
        "factors": factors
    }

def perform_complete_image_analysis(image, classifier):
    """Perform comprehensive image analysis"""
    results = {}
    
    with st.spinner("🔍 Analyzing facial emotions with DeepFace..."):
        deepface_result = analyze_with_deepface(image)
    
    with st.spinner("🔍 Running secondary emotion analysis..."):
        hf_result = analyze_with_huggingface(image, classifier)
    
    combined_result = combine_emotion_analyses(deepface_result, hf_result)
    results['facial_emotions'] = combined_result
    
    with st.spinner("🎨 Performing color psychology analysis..."):
        results['color_analysis'] = analyze_colors(image)
    
    with st.spinner("📐 Analyzing composition and lighting..."):
        results['composition'] = analyze_composition(image)
    
    results['overall_sentiment'] = calculate_overall_sentiment(results)
    
    return results

# ==================== LOAD MODELS ====================

with st.spinner("🔄 Loading AI Models... (First run may take 1-2 minutes)"):
    unified_model = load_text_models()
    image_classifier = load_image_models()

if unified_model:
    st.success("✅ Text model loaded successfully! (Using unified model for both branches)")
if image_classifier:
    st.success("✅ Image models loaded successfully!")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("⚙️ Analysis Mode")
    
    analysis_mode = st.radio(
        "Select Mode:",
        ["Single Text", "Batch Analysis", "Image Analysis", "History"]
    )
    
    st.markdown("---")
    
    if analysis_mode in ["Single Text", "Batch Analysis"]:
        st.markdown("### 🎛️ Text Analysis Settings")
        
        combination_strategy = st.selectbox(
            "Combination Strategy:",
            ["Score Difference", "Weighted Average", "Confidence Voting", "Threshold-Based", "Neural Fusion"]
        )
        
        neutral_threshold = st.slider("Neutral Zone:", 0.0, 0.5, 0.2, 0.05)
        positive_weight = st.slider("Positive Weight:", 0.0, 1.0, 0.5, 0.1)
        apply_penalties = st.checkbox("Apply Content Penalties", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.info("""
    **Text Analysis:**
    - Unified model architecture
    - Same model for both branches
    - Positive & Negative views
    
    **Image Analysis:**
    - DeepFace + HuggingFace
    - Color psychology
    - Composition analysis
    """)

# ==================== MAIN INTERFACE ====================

if analysis_mode == "Single Text":
    st.header("🔍 Single Text Analysis")
    
    with st.expander("📝 Try Test Cases"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Positive", use_container_width=True):
                st.session_state.example_text = "This is absolutely amazing! I love it!"
        with col2:
            if st.button("Negative", use_container_width=True):
                st.session_state.example_text = "This is terrible! I hate it!"
        with col3:
            if st.button("Subtle", use_container_width=True):
                st.session_state.example_text = "It's okay, not great but not bad either"
    
    default_text = st.session_state.get('example_text', '')
    user_input = st.text_area("Enter text:", value=default_text, height=120)
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            if 'example_text' in st.session_state:
                del st.session_state.example_text
            st.rerun()
    
    if analyze_btn and user_input:
        with st.spinner("🔄 Processing..."):
            result = analyze_text_dual_branch(
                user_input, unified_model,
                combination_strategy, neutral_threshold, positive_weight, apply_penalties
            )
            
            if result:
                st.session_state.text_history.append({
                    'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                    'sentiment': result['label'],
                    'final_score': result['final_score'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main result box
                warnings = ""
                if result['features']['has_violence']:
                    warnings += " 🚨 Violent Content"
                if abs(result['score_difference']) < 0.1:
                    warnings += " ⚠️ Close Call"
                
                st.markdown(
                    f'<div class="sentiment-box" style="background-color: {result["color"]};">'
                    f'<h1 style="text-align: center; margin: 0;">{result["sentiment"]}{warnings}</h1>'
                    f'<p style="text-align: center; font-size: 1.2rem; margin-top: 0.5rem;">'
                    f'Final Score: {result["final_score"]:.3f} | Confidence: {result["confidence"]:.2%}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Branch outputs
                st.markdown("### 🔬 Individual Branch Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Positive Detection Branch")
                    st.markdown(
                        f'<div class="model-branch" style="border-color: #4CAF50; background-color: #4CAF5020;">'
                        f'<div class="score-display" style="color: #4CAF50;">{result["positive_score"]:.3f}</div>'
                        f'<p style="text-align: center;"><strong>Positive Confidence</strong></p>'
                        f'<p style="text-align: center;">Raw: {result["positive_branch"]["interpretation"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown("#### 📉 Negative Detection Branch")
                    st.markdown(
                        f'<div class="model-branch" style="border-color: #F44336; background-color: #F4433620;">'
                        f'<div class="score-display" style="color: #F44336;">{result["negative_score"]:.3f}</div>'
                        f'<p style="text-align: center;"><strong>Negative Confidence</strong></p>'
                        f'<p style="text-align: center;">Raw: {result["negative_branch"]["interpretation"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Visualizations
                st.markdown("### 📊 Visual Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Dual branch visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Positive Branch',
                        x=['Positive Detector'],
                        y=[result['positive_score']],
                        marker_color='#4CAF50',
                        text=[f"{result['positive_score']:.3f}"],
                        textposition='auto',
                        width=0.4
                    ))
                    fig.add_trace(go.Bar(
                        name='Negative Detector',
                        x=['Negative Detector'],
                        y=[result['negative_score']],
                        marker_color='#F44336',
                        text=[f"{result['negative_score']:.3f}"],
                        textposition='auto',
                        width=0.4
                    ))
                    fig.update_layout(
                        title="Model Outputs (Unified Model)",
                        yaxis_title="Confidence Score",
                        yaxis=dict(range=[0, 1]),
                        height=350,
                        showlegend=True,
                        barmode='group',
                        annotations=[
                            dict(
                                text="© Priyanshu Joarder",
                                xref="paper", yref="paper",
                                x=1, y=-0.25,
                                showarrow=False,
                                font=dict(size=10, color="gray"),
                                xanchor='right'
                            )
                        ]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Score comparison
                    categories = ['Positive\nBranch', 'Negative\nBranch', 'Final\nScore']
                    values = [result['positive_score'], result['negative_score'], result['final_score']]
                    colors = ['#4CAF50', '#F44336', result['color']]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=categories,
                            y=values,
                            marker_color=colors,
                            text=[f"{v:.3f}" for v in values],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Score Comparison",
                        yaxis_title="Score",
                        yaxis=dict(range=[-1, 1]),
                        height=350,
                        annotations=[
                            dict(
                                text="© Priyanshu Joarder",
                                xref="paper", yref="paper",
                                x=1, y=-0.25,
                                showarrow=False,
                                font=dict(size=10, color="gray"),
                                xanchor='right'
                            )
                        ]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Fusion gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['final_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Final Fused Score", 'font': {'size': 24}},
                    number={'font': {'size': 50}},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [-1, 1], 'tickwidth': 2},
                        'bar': {'color': "darkblue", 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [-1, -0.3], 'color': '#FFB6C6'},
                            {'range': [-0.3, 0.3], 'color': '#FFE5B4'},
                            {'range': [0.3, 1], 'color': '#90EE90'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.8,
                            'value': result['final_score']
                        }
                    }
                ))
                fig.update_layout(
                    height=400,
                    annotations=[
                        dict(
                            text="© Priyanshu Joarder",
                            xref="paper", yref="paper",
                            x=1, y=-0.15,
                            showarrow=False,
                            font=dict(size=10, color="gray"),
                            xanchor='right'
                        )
                    ]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics
                st.markdown("### 📋 Detailed Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Label", result['label'])
                with col2:
                    st.metric("Score Difference", f"{result['score_difference']:.3f}")
                with col3:
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                with col4:
                    st.metric("Strategy", combination_strategy.split()[0])
                
                # Content features
                st.markdown("### 🔍 Content Analysis")
                
                features = result['features']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Positive Words", features['positive_count'])
                with col2:
                    st.metric("Negative Words", features['negative_count'])
                with col3:
                    st.metric("Violent Words", features['violent_count'])
                with col4:
                    st.metric("Negations", features['negation_count'])
                
                # Interpretation
                st.markdown("### 💡 System Interpretation")
                
                interpretation = f"""
**Dual-Branch Analysis Complete (Unified Model):**

- **Positive Branch Score**: {result['positive_score']:.3f} - {"High" if result['positive_score'] > 0.6 else "Moderate" if result['positive_score'] > 0.3 else "Low"} positive sentiment detected
- **Negative Branch Score**: {result['negative_score']:.3f} - {"High" if result['negative_score'] > 0.6 else "Moderate" if result['negative_score'] > 0.3 else "Low"} negative sentiment detected
- **Score Difference**: {result['score_difference']:.3f} ({'Positive dominates' if result['score_difference'] > 0.2 else 'Negative dominates' if result['score_difference'] < -0.2 else 'Closely matched'})

**Fusion Strategy**: {combination_strategy}
- The system combined both branches using {combination_strategy.lower()} method
- Final score: {result['final_score']:.3f} → Classified as **{result['label']}**
"""
                
                # Add special case explanation for neutral/subtle text
                if result['label'] == 'NEUTRAL':
                    if result['positive_score'] < 0.35 and result['negative_score'] < 0.35:
                        interpretation += "\n⚖️ **Neutral Classification**: Both positive and negative scores are low, indicating genuinely neutral or subtle sentiment.\n"
                    else:
                        interpretation += "\n⚖️ **Neutral Classification**: Positive and negative sentiments are balanced, resulting in neutral classification.\n"
                
                if features['has_violence']:
                    interpretation += "\n🚨 **Violence Warning**: Threatening language detected. Positive score was reduced by 80%.\n"
                
                if abs(result['score_difference']) < 0.1 and result['label'] != 'NEUTRAL':
                    interpretation += "\n⚠️ **Close Decision**: Both branches gave similar scores. Result may be ambiguous.\n"
                
                # Only show negation warning if there are 3+ negations (more likely to be problematic)
                if features['negation_count'] >= 2 and result['label'] != 'NEUTRAL':
                    interpretation += "\n🔄 **Negation Detected**: Multiple negations found - sentiment interpretation may be complex.\n"
                
                st.markdown(interpretation)

elif analysis_mode == "Batch Analysis":
    st.header("📚 Batch Text Analysis")
    
    batch_input = st.text_area("Enter texts (one per line):", height=200)
    
    if st.button("🚀 Analyze All", type="primary"):
        if batch_input:
            texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
            
            with st.spinner(f"Processing {len(texts)} texts..."):
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts):
                    result = analyze_text_dual_branch(
                        text, unified_model,
                        combination_strategy, neutral_threshold, positive_weight, apply_penalties
                    )
                    if result:
                        results.append({
                            'Text': text[:60] + "..." if len(text) > 60 else text,
                            'Label': result['label'],
                            'Score': round(result['final_score'], 3)
                        })
                    progress_bar.progress((i + 1) / len(texts))
                
                progress_bar.empty()
            
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", len([r for r in results if r['Label'] == 'POSITIVE']))
                with col2:
                    st.metric("Negative", len([r for r in results if r['Label'] == 'NEGATIVE']))
                with col3:
                    st.metric("Neutral", len([r for r in results if r['Label'] == 'NEUTRAL']))

elif analysis_mode == "Image Analysis":
    st.header("🖼️ Image Sentiment Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("⏳ Analyzing..."):
                    try:
                        result = perform_complete_image_analysis(image, image_classifier)
                        st.session_state.image_analysis_result = result
                        st.success("✅ Complete!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    with col2:
        st.markdown("### 📊 Results")
        
        if st.session_state.image_analysis_result:
            result = st.session_state.image_analysis_result
            overall = result['overall_sentiment']
            
            sentiment_emojis = {'positive': '😊', 'negative': '😔', 'neutral': '😐'}
            sentiment_colors = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
            
            color = sentiment_colors[overall['sentiment']]
            emoji = sentiment_emojis[overall['sentiment']]
            
            st.markdown(f"""
                <div style='background: white; padding: 30px; border-radius: 15px; text-align: center; 
                            box-shadow: 0 8px 16px rgba(0,0,0,0.15); border: 3px solid {color};'>
                    <h1 style='font-size: 80px; margin: 0;'>{emoji}</h1>
                    <h2 style='color: {color}; margin: 15px 0; text-transform: uppercase;'>
                        {overall['sentiment']}
                    </h2>
                    <p style='font-size: 28px; color: #64748b; font-weight: bold;'>
                        {overall['confidence']}% Confident
                    </p>
                    <p style='color: #64748b; margin-top: 15px;'>{overall['description']}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Upload and analyze an image")
    
    # Detailed Results
    if st.session_state.image_analysis_result:
        result = st.session_state.image_analysis_result
        
        st.markdown("---")
        
        if result['overall_sentiment']['factors']:
            st.markdown("### 🔍 Key Factors")
            for i, factor in enumerate(result['overall_sentiment']['factors'], 1):
                st.markdown(f"{i}. {factor}")
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "😊 Facial Analysis", 
            "🎨 Color Psychology", 
            "📐 Composition",
            "📊 Visualizations"
        ])
        
        with tab1:
            if result['facial_emotions']['detected']:
                emotions = result['facial_emotions']['emotions']
                dominant = result['facial_emotions']['dominant_emotion']
                conf = result['facial_emotions'].get('confidence', 'N/A')
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.success(f"✅ **Dominant:** {dominant.upper()}")
                    st.metric("Confidence", f"{conf}%")
                    
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    
                    for emotion, score in sorted_emotions:
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            st.write(f"**{emotion.capitalize()}**")
                        with col_b:
                            st.progress(float(score) / 100.0)
                            st.caption(f"{score:.1f}%")
                
                with col2:
                    if 'age' in result['facial_emotions']:
                        st.metric("Age", result['facial_emotions']['age'])
                    if 'gender' in result['facial_emotions']:
                        st.metric("Gender", result['facial_emotions']['gender'])
            else:
                st.warning("⚠️ No face detected")
        
        with tab2:
            color_data = result['color_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Color Metrics")
                st.metric("Brightness", f"{color_data['brightness']}/255")
                st.metric("Saturation", f"{color_data['saturation']}/255")
                
                rgb = color_data['rgb_values']
                st.markdown("### RGB Values")
                st.write(f"🔴 Red: {rgb['R']}")
                st.write(f"🟢 Green: {rgb['G']}")
                st.write(f"🔵 Blue: {rgb['B']}")
            
            with col2:
                st.markdown("### Emotional Impact")
                st.info(f"**Temperature:** {color_data['temperature']}")
                st.success(f"**Impact:** {color_data['emotional_impact']}")
                
                fig = go.Figure(data=[go.Bar(
                    x=['Red', 'Green', 'Blue'],
                    y=[rgb['R'], rgb['G'], rgb['B']],
                    marker_color=['#FF6B6B', '#51CF66', '#4DABF7'],
                    text=[rgb['R'], rgb['G'], rgb['B']],
                    textposition='auto',
                )])
                fig.update_layout(
                    title="RGB Distribution",
                    height=300,
                    showlegend=False,
                    yaxis_range=[0, 255],
                    annotations=[
                        dict(
                            text="© Priyanshu Joarder",
                            xref="paper", yref="paper",
                            x=1, y=-0.3,
                            showarrow=False,
                            font=dict(size=10, color="gray"),
                            xanchor='right'
                        )
                    ]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            comp_data = result['composition']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Contrast", f"{comp_data['contrast']:.1f}")
                st.caption(comp_data['contrast_sentiment'])
            
            with col2:
                st.metric("Brightness", f"{comp_data['brightness_mean']:.1f}")
                st.caption(comp_data['brightness_sentiment'])
            
            with col3:
                st.metric("Edge Density", f"{comp_data['edge_density']:.1f}%")
                st.caption(comp_data['complexity'])
            
            st.markdown("---")
            st.write(f"""
            **Summary:** This image has **{comp_data['complexity']}** composition with 
            **{comp_data['contrast_sentiment']}** contrast and 
            **{comp_data['brightness_sentiment']}** lighting.
            """)
        
        with tab4:
            st.markdown("### 📈 Emotion Distribution")
            
            if result['facial_emotions']['detected']:
                emotions_df = pd.DataFrame(
                    list(result['facial_emotions']['emotions'].items()),
                    columns=['Emotion', 'Score']
                ).sort_values('Score', ascending=False)
                
                fig = px.bar(
                    emotions_df, 
                    x='Emotion', 
                    y='Score',
                    color='Score',
                    color_continuous_scale='Viridis',
                    title='Detected Emotions',
                    text='Score'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    annotations=[
                        dict(
                            text="© Priyanshu Joarder",
                            xref="paper", yref="paper",
                            x=1, y=-0.18,
                            showarrow=False,
                            font=dict(size=10, color="gray"),
                            xanchor='right'
                        )
                    ]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🎯 Sentiment Score")
            overall_score = result['overall_sentiment']['score']
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Score", 'font': {'size': 24}},
                delta={'reference': 0, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [-100, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue", 'thickness': 0.3},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [-100, -25], 'color': '#fee2e2'},
                        {'range': [-25, 25], 'color': '#fef3c7'},
                        {'range': [25, 100], 'color': '#d1fae5'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': overall_score
                    }
                }
            ))
            fig.update_layout(
                height=400,
                font={'size': 16},
                annotations=[
                    dict(
                        text="© Priyanshu Joarder",
                        xref="paper", yref="paper",
                        x=1, y=-0.15,
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        xanchor='right'
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)

else:  # History
    st.header("📜 Analysis History")
    
    if st.session_state.text_history:
        df = pd.DataFrame(st.session_state.text_history)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("### 📊 Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Analyses", len(st.session_state.text_history))
        with col2:
            avg_score = np.mean([h['final_score'] for h in st.session_state.text_history])
            st.metric("Average Score", f"{avg_score:.3f}")
        with col3:
            pos_count = len([h for h in st.session_state.text_history if h['sentiment'] == 'POSITIVE'])
            st.metric("Positive Count", pos_count)
        
        # Distribution
        sentiment_counts = {}
        for h in st.session_state.text_history:
            sentiment_counts[h['sentiment']] = sentiment_counts.get(h['sentiment'], 0) + 1
        
        if sentiment_counts:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(sentiment_counts.keys()),
                    values=list(sentiment_counts.values()),
                    marker=dict(colors=['#90EE90', '#FFB6C6', '#FFE5B4'])
                )
            ])
            fig.update_layout(
                title="Sentiment Distribution",
                height=400,
                annotations=[
                    dict(
                        text="© Priyanshu Joarder",
                        xref="paper", yref="paper",
                        x=1, y=-0.15,
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        xanchor='right'
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.text_history = []
                st.rerun()
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="sentiment_history.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("📝 No history yet. Start analyzing!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: blue; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;'>
        <h3>🏗️ Multimodal Sentiment Analysis System</h3>
        <p><strong>Text:</strong> Unified Model Architecture (Same model for both branches)</p>
        <p><strong>Image:</strong> Multi-Model Fusion (DeepFace + HuggingFace + Psychology)</p>
    </div>
""", unsafe_allow_html=True)