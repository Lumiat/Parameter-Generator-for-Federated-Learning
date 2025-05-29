import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, image_size, num_classes, in_channels):
        super(CNN, self).__init__()
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size = 5, padding=2), 
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2)

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 2 * 16, kernel_size = 3), 
            torch.nn.BatchNorm2d(16 * 2),
            torch.nn.ReLU(),
        )
        self.conv3=torch.nn.Sequential(
            torch.nn.Conv2d(2 * 16, 4 * 16, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2=torch.nn.MaxPool2d(kernel_size = 2)
        self.fc = torch.nn.Linear(5 * 5 * 4 * 16, num_classes)
        
    
    # forward propagation
    def forward(self,x):
        out = self.pool1(self.conv1(x))
        out = self.pool2(self.conv3(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
