import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CAMAttentionLayer(nn.Module):
    def __init__(self, channel):
        super(CAMAttentionLayer, self).__init__()
        # Transform 1-channel CAM to match feature channels
        self.cam_expand = nn.Conv2d(1, channel, kernel_size=1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()  # This will generate weights between 0 and 1
        )

    def forward(self, x, cams):
        # Resize CAMs to match feature maps size
        cams_resized = F.interpolate(cams, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # Expand CAM channels
        cams_expanded = self.cam_expand(cams_resized)
        # Generate attention weights from CAMs
        cam_weights = self.channel_attention(cams_expanded)
        # Apply attention by element-wise multiplication of weights and feature maps
        x = x * cam_weights
        return x


class EfficientNetB4WithCAMAttention(nn.Module):
    def __init__(self, num_classes, device=torch.device('cpu')):
        super(EfficientNetB4WithCAMAttention, self).__init__()
        self.base_model = models.efficientnet_b4(pretrained=False)
        self.base_model.load_state_dict(torch.load('models/efficientnet_b4_rwightman-7eb33cd5.pth', map_location=device))
        self.base_model.classifier = nn.Identity()

        # Assume the output feature map has a certain number of channels
        out_channels = self.base_model.features[-1].out_channels
        self.cam_attention = CAMAttentionLayer(out_channels)
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x, cams):
        # Extract features from the base model
        features = self.base_model.features(x)

        # Apply CAM-based attention
        features = self.cam_attention(features, cams)

        # Global average pooling and classification
        x = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x