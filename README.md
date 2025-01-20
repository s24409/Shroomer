# Fungi Classifier App

A web application for classifying Danish fungi species using deep learning.

## Tech Stack

### Backend
- **Framework**: Flask 2.0.1 (Python web framework)
- **Python Version**: 3.9
- **Machine Learning**:
  - PyTorch 2.1.2 + Torchvision 0.16.2 (Deep Learning framework)
  - Timm 0.9.12 (PyTorch Image Models)
  - Model Architecture: ViT (Vision Transformer) base patch16 224
  - Albumentations 1.3.1 (Image augmentation)
  - OpenCV 4.8.1 (Image processing)

### Frontend
- **HTML/CSS/JavaScript** (Vanilla)
- Simple and responsive UI for image upload and classification
- Real-time image preview
- Asynchronous API calls for classification

### Model Details
- Base model: vit_base_patch16_224
- Input size: 224x224 pixels
- Number of classes: 1604
- Preprocessing: Resize, Normalize (ImageNet stats)
- Output: Class prediction and confidence score

## Project Structure
```
fungi-classifier-app/
├── app/
│   ├── __init__.py          # Flask app initialization
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   └── classifier.py    # Model inference logic
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── utils/
│   │   ├── preprocessing.py # Image preprocessing
│   │   └── validation.py    # Input validation
│   ├── static/
│   │   ├── css/style.css   # UI styling
│   │   └── js/main.js      # Frontend logic
│   └── templates/
│       └── index.html      # Web interface
├── docker/
│   └── Dockerfile          # Container configuration
├── scripts/
│   └── download_model.sh   # Model download utility
└── requirements.txt        # Python dependencies
```

## Running the application

The application supports two modes of operation:

### 1. API Mode
- Uses HuggingFace's inference API
- Requires HF_API_TOKEN
- Reduced deployment size
- No need to download model weights

### 2. Local Mode
- Runs model inference locally
- Requires downloading model weights
- Direct PyTorch inference
- Better for production deployment

To run the application:

1. Create a Python virtual environment
- using Conda
```bash
conda create -n fungi python=3.9
conda activate fungi
# conda install --file requirements.txt
pip install -r requirements.txt
```
- or using venv
```bash
python -m venv fungi-env
source fungi-env/bin/activate  # On Windows use `fungi-env\Scripts\activate`
pip install -r requirements.txt
```

2. Download the model weights:
```bash
# Install required packages
pip install requests tqdm

# Run the download script
python scripts/download_model.py
```

3. Set environment variables
```bash
cd fungi-classifier-app
cp .env.template .env
```
And replace the values with yours.

### Local vs API mode
The `MODEL_TYPE` envvar controls the way inference in the application works:
- For 'local' the weights in `/app/models/weights/exp1/best_f1.pth` will be used, which can be changed in the code.
- For 'api' a valid Hugging Face API key is needed and a model_id with handler.py defined

### Environment Variables
- `SECRET_KEY`: Signing Session key (also other, like CSRF protection when implemented) 
- `FLASK_APP`: Application entry point
- `FLASK_ENV`: Development/Production mode
- `MODEL_TYPE`: Local/API inference mode
- `HF_API_TOKEN`: Hugging Face API token (for API mode)
- `HF_MODEL_ID`: Hugging Face model identifier
- `MODEL_PATH`: Local model weights path


4. Run the application
```bash
flask run
```

For containerized deployment:
(download weights first)
```bash
docker build -t fungi-classifier -f Dockerfile .
docker run -p 5000:5000 fungi-classifier
```

## Features
- Image upload and preview
- Real-time fungi classification
- Support for both local and API-based inference
- Configurable model deployment options
- Docker containerization for easy deployment
- Error handling and input validation
- Support for multiple image formats (JPG, JPEG, PNG)

## Dependencies
Key dependencies are managed through requirements.txt and include:
- Flask and related web dependencies
- PyTorch ecosystem for ML
- Image processing libraries
- Development and utility tools

For the complete list of dependencies, see requirements.txt.
````
