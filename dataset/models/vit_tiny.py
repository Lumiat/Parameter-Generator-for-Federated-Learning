import torch.nn as nn
import timm

class VitTiny(nn.Module):
    def __init__(self, image_size, num_classes, in_channels):
        super(VitTiny, self).__init__()
        self.model = timm.create_model(
            "vit_tiny_patch16_224", 
            pretrained=False,
            img_size = (image_size, image_size),
            patch_size = int(image_size // 4) if image_size < 224 else 16,
            in_chans = in_channels,
            num_classes = num_classes
        )

    # # patch embed layer customize to in_channels    
    # self.model.patch_embed.proj = nn.Conv2d(
    #         in_channels,
    #         self.model.embed_dim, 
    #         kernel_size=self.model.patch_embed.patch_size,
    #         stride=self.model.patch_embed.patch_size
    #     )
    # nn.init.kaiming_normal_(self.model.patch_embed.proj.weight)

    # # classifier head customize to num_classes
    # self.fc = nn.Linear(self.model.embed_dim, num_classes)
    
    def forward(self, x):
        return self.model(x)
