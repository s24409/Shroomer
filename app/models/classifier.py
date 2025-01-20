from app.utils.species_lookup import SpeciesLookup
from abc import ABC, abstractmethod
import torch
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import requests
import os

class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, image):
        pass

class HFAPIClassifier(BaseClassifier):
    def __init__(self, model_id, api_token):
        self.model_id = model_id
        self.api_token = api_token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def predict(self, image):
        with open(image, "rb") as f:
            data = f.read()
        response = requests.post(self.api_url, headers=self.headers, data=data)
        print(response.json())
        return response.json()

class LocalClassifier(BaseClassifier):
    def __init__(self, model_path, model_file: str = "best_f1.pth"):
        self.species_lookup = SpeciesLookup()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1604)
        
        state_dict = torch.load(os.path.join(model_path, model_file), map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def predict(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        pred_idx = probs[0].argmax().item()
        confidence = probs[0].max().item()
        
        species_info = self.species_lookup.get_species_info(pred_idx)
        
        return {
            "species_info": species_info,
            "score": confidence
        }

def get_classifier(config):
    if config['MODEL_TYPE'] == 'api':
        if not config['HF_API_TOKEN']:
            raise ValueError("HF_API_TOKEN is required for API classifier")
        return HFAPIClassifier(config['HF_MODEL_ID'], config['HF_API_TOKEN'])
    else:
        if not os.path.exists(config['MODEL_PATH']):
            raise ValueError(f"Model not found at {config['MODEL_PATH'], config['MODEL_FILE']}")
        return LocalClassifier(config['MODEL_PATH'])
