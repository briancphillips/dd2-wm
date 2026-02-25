import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    """
    Standard ResNet18 model modified to handle either 32x32 (CIFAR/GTSRB) 
    or 224x224 (VGGFace2/CheXpert) images.
    """
    def __init__(self, num_classes=100, is_32x32=True):
        super(ResNet18, self).__init__()
        
        # Load standard PyTorch ResNet18
        self.model = models.resnet18(weights=None)
        
        if is_32x32:
            # Modify the first convolutional layer to handle 32x32 inputs
            # Standard ResNet uses 7x7 conv with stride 2, which is too aggressive for 32x32
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            # Remove the maxpool layer (again, too aggressive downsampling for 32x32)
            self.model.maxpool = nn.Identity()
        
        # Modify the final fully connected layer for the target number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        """
        Extracts features from the penultimate layer (before the FC layer).
        Useful for Mahalanobis distance scoring in DynaDetect.
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
