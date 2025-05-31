import timm
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, image_size, num_classes, in_channels):
        super(ResNet18, self).__init__()
        self.model = timm.create_model(
            "resnet18",
            pretrained=False,
            in_chans = in_channels,
            num_classes = num_classes,
        )

    def forward(self, x):
        return self.model(x)
    

# import torch.nn as nn
# import timm


# def Model():
#     model = timm.create_model("resnet18", pretrained=True)
#     model.fc = nn.Linear(512, 10)
#     return model


# if __name__ == "__main__":
#     model, _ = Model()
#     print(model)
#     num_param = 0
#     for v in model.parameters():
#         num_param += v.numel()
#     print("num_param:", num_param)

# import torch.nn as nn
# import torchvision.models as models


# class ResNet18(nn.Module):
#     def __init__(self, image_size, num_classes, in_channels):
#         super(ResNet18, self).__init__()
#         self.model = models.resnet18(weights=None)

#         original_first_conv = self.model.conv1
#         self.model.conv1 = nn.Conv2d(
#             in_channels, 
#             original_first_conv.out_channels,
#             kernel_size=original_first_conv.kernel_size if image_size >= 224 else 3,
#             stride=original_first_conv.stride if image_size >= 224 else 1,
#             padding=original_first_conv.padding,
#             bias=original_first_conv.bias
#         )

#         if(image_size < 244):
#             self.model.maxpool = nn.Identity()

#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  

#     def forward(self, x):
#         return self.model(x)
