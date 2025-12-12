# 🎭 Multimodal Sentiment Analysis System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

An advanced ML-based sentiment analysis application that performs comprehensive analysis on both text and images using state-of-the-art deep learning models.


## ✨ Features

### 📝 Text Analysis
- **Dual-Branch Architecture**: Uses unified BERT model for both positive and negative sentiment detection
- **Multiple Combination Strategies**: 
  - Score Difference
  - Weighted Average
  - Confidence Voting
  - Threshold-Based
  - Neural Fusion
- **Content Feature Detection**: Identifies positive/negative words, violence, negations
- **Batch Processing**: Analyze multiple texts simultaneously
- **Interactive Visualizations**: Real-time charts and gauges

### 🖼️ Image Analysis
- **Facial Emotion Detection**: Using DeepFace with multiple backend options
- **Secondary Analysis**: HuggingFace emotion classification
- **Color Psychology**: Analyzes emotional impact of colors
- **Composition Analysis**: Evaluates contrast, brightness, and complexity
- **Multi-Model Fusion**: Combines multiple models for robust predictions

### 📊 Additional Features
- Analysis history tracking
- Downloadable CSV reports
- Interactive Plotly visualizations
- Real-time confidence scores
- Customizable analysis parameters

## 🚀 Demo

Try the live demo: [Link to deployed app]

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU support optional (for faster processing)

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/multimodal-sentiment-analysis.git
cd multimodal-sentiment-analysis
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: First-time setup may take 5-10 minutes as it downloads pre-trained models (~1-2GB).

### 4. Install system dependencies (for DeepFace)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install opencv
```

**Windows:**
No additional steps required.

## 🎮 Usage

### Run the application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Basic Workflow

#### Text Analysis:
1. Select "Single Text" mode from the sidebar
2. Enter your text or try a test case
3. Adjust analysis settings (optional)
4. Click "🚀 Analyze Text"
5. View detailed results and visualizations

#### Image Analysis:
1. Select "Image Analysis" mode
2. Upload an image (JPG, PNG, WEBP)
3. Click "🚀 Analyze Image"
4. Explore facial emotions, color psychology, and composition analysis

#### Batch Analysis:
1. Select "Batch Analysis" mode
2. Enter multiple texts (one per line)
3. Click "🚀 Analyze All"
4. View aggregated results and statistics

## 📁 Project Structure

```
multimodal-sentiment-analysis/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
├── LICENSE                    # MIT License
├── README.md                  # This file
│
├── assets/                    # Screenshots and images
│   └── screenshot.png
│
├── docs/                      # Documentation
│   ├── INSTALLATION.md        # Detailed installation guide
│   ├── USAGE.md              # User guide
│   └── API.md                # API documentation
│
└── tests/                     # Test files (optional)
    └── test_app.py
```

## 🔬 Models Used

### Text Analysis
- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Architecture**: BERT-based multilingual sentiment classifier
- **Languages**: Supports 6+ languages
- **Output**: 5-star rating system

### Image Analysis
- **DeepFace**: Face detection and emotion recognition
  - Backends: RetinaFace, MTCNN, OpenCV, SSD
  - Actions: Emotion, age, gender, race detection
- **HuggingFace**: `dima806/facial_emotions_image_detection`
  - Secondary emotion validation
  - 7 emotion categories

## ⚙️ Configuration

### Analysis Settings

Customize analysis behavior through the sidebar:

- **Combination Strategy**: Choose how positive/negative scores are combined
- **Neutral Zone Threshold**: Adjust sensitivity (0.0 - 0.5)
- **Positive Weight**: Balance between positive and negative detection
- **Content Penalties**: Enable/disable context-aware adjustments

### Environment Variables

Create a `.env` file for custom configurations:

```env
# Model cache directory
TRANSFORMERS_CACHE=./models/cache

# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# TensorFlow logging
TF_CPP_MIN_LOG_LEVEL=2
```

## 📊 Example Results

### Text Analysis Output
```
Input: "This is absolutely amazing! I love it!"

Results:
- Sentiment: POSITIVE 😊
- Final Score: 0.847
- Confidence: 92%
- Positive Branch: 0.923
- Negative Branch: 0.076
```

### Image Analysis Output
```
Facial Emotions:
- Happy: 85%
- Neutral: 10%
- Surprise: 5%

Overall Sentiment: POSITIVE (92% confident)
Color Temperature: Warm and energetic
Composition: Bright and cheerful
```


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Priyanshu Joarder**

- GitHub: https://github.com/Priyanshu-4096
- LinkedIn: https://www.linkedin.com/in/priyanshu-joarder-308724144/

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Web framework
- [Hugging Face](https://huggingface.co/) - Pre-trained models
- [DeepFace](https://github.com/serengil/deepface) - Face analysis library
- [Plotly](https://plotly.com/) - Interactive visualizations


<div align="center">
Made by Priyanshu Joarder
</div>
