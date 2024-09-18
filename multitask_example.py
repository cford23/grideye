import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign


# Define a simple object detection model
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, num_classes * 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        bbox_preds = self.conv2(x)
        class_preds = self.conv3(x)
        return bbox_preds, class_preds
    
class MultiCategoryModel(nn.Module):
    def __init__(self, num_boundary_classes, num_character_classes):
        super(MultiCategoryModel, self).__init__()

        # Use a pre-trained ResNet backbone for feature extraction
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # Remove the fully connected layers

        # Define region of interest (RoI) pooling (optional)
        self.roi_pool = MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

        # Detection heads for boundaries and characters
        self.boundary_head = DetectionHead(in_channels=2048, num_classes=num_boundary_classes)
        self.character_head = DetectionHead(in_channels=2048, num_classes=num_character_classes)

    def forward(self, x):
        # Extract shared features from the backbone
        features = self.backbone(x)

        # Apply category-specific heads
        boundary_bbox_preds, boundary_class_preds = self.boundary_head(features)
        character_bbox_preds, character_class_preds = self.character_head(features)

        return {
            'boundary': {
                'bbox_preds': boundary_bbox_preds,
                'class_preds': boundary_class_preds
            },
            'character': {
                'bbox_preds': character_bbox_preds,
                'class_preds': character_class_preds
            }
        }


if __name__ == '__main__':
    boundary_classes = ['tree', 'building', 'wall']
    character_classes = ['knight', 'archer', 'mage']

    num_boundary_classes = len(boundary_classes)
    num_character_classes = len(character_classes)

    model = MultiCategoryModel(
        num_boundary_classes=num_boundary_classes,
        num_character_classes=num_character_classes
    )

    # Generate dummy input (batch size of 2, 3 channels, 224x224 image)
    dummy_input = torch.randn(2, 3, 224, 224)

    # Forward pass through the model
    output = model(dummy_input)

    # Print output shapes
    print("Boundary bbox predictions shape:", output['boundary']['bbox_preds'].shape)
    print("Boundary class predictions shape:", output['boundary']['class_preds'].shape)
    print("Character bbox predictions shape:", output['character']['bbox_preds'].shape)
    print("Character class predictions shape:", output['character']['class_preds'].shape)
