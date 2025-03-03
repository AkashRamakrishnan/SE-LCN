import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Custom ELayer definition
class ELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EfficientNetB4WithELayer(nn.Module):
    def __init__(self, num_classes, cam_channels, device=torch.device('cpu')):
        super(EfficientNetB4WithELayer, self).__init__()
        self.base_model = models.efficientnet_b4(pretrained=False)
        self.base_model.load_state_dict(torch.load('models/efficientnet_b4_rwightman-7eb33cd5.pth', map_location=device))
        self.base_model.classifier = nn.Identity()

        # Assume the output feature map has a certain number of channels
        out_channels = self.base_model.features[-1].out_channels
        self.e_layer = ELayer(out_channels + cam_channels, out_channels)
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x, cams):
        # Extract features from the base model
        features = self.base_model.features(x)

        # Resize CAMs to match the feature maps' spatial dimensions
        cams_resized = F.interpolate(cams, size=(features.size(2), features.size(3)), mode='bilinear', align_corners=False)

        # Concatenate the feature maps with the resized CAMs
        x = torch.cat([features, cams_resized], dim=1)
        
        # Apply the E-layer
        x = self.e_layer(x)
        
        # Global average pooling and classification
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x