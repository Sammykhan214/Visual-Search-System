import torch
import torch.nn as nn
from torchvision import transforms, models
import timm

class FeatureExtractorBase:
    def __init__(self, device='cpu'):
        self.device = device
        self.model= None
        self.preprocess = None
        self.embedding_dim = None
    
    def extract_batch(self, image_tensors):
        raise NotImplementedError
    
class ResNetExtractor(FeatureExtractorBase):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # Remove final classification layer
        self.model = self.model.to(device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.embedding_dim = 512
    
    def extract_batch(self, image_tensors):
        with torch.no_grad():
            return self.model(image_tensors.to(self.device))

class MobileNetV3Extractor(FeatureExtractorBase):
    def __init__(self, device='cpu', variant='small'):
        super().__init__(device)
        model_name = f'mobilenetv3_{variant}_100'
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 to get features
        self.model = self.model.to(device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.embedding_dim = self.model.num_features

    def extract_batch(self, image_tensors):
        with torch.no_grad():
            return self.model(image_tensors.to(self.device))

def get_extractor(model_name: str, device='cuda') -> FeatureExtractorBase:
    if model_name == 'resnet50':
        return ResNetExtractor(device)
    elif model_name == ('mobilenetv3'):
        return MobileNetV3Extractor(device, 'small')
    elif model_name == 'mobilenetv3_large':
        return MobileNetV3Extractor(device, 'large')
    else:
        raise ValueError(f"Unsupported model name: {model_name}")