import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    
    # Model settings
    MODEL_TYPE = os.environ.get('MODEL_TYPE', 'api')  # 'api' or 'local'
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/fungi-classifier')
    MODEL_FILE = os.environ.get('MODEL_FILE', 'best_f1.pth')
    HF_API_TOKEN = os.environ.get('HF_API_TOKEN')
    HF_MODEL_ID = os.environ.get('HF_MODEL_ID', '70ziko/fungi-classifier')
    
    # Image settings
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    
    @staticmethod
    def init_app(app):
        # Create upload directory if it doesn't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
