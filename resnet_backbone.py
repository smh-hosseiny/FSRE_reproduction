import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class ResNetMultiImageInput(models.ResNet):
    
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # super().__init__(block, layers)
        # self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # for module in self.modules():
        #     if isinstance(module, nn.Conv2d):
        #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(module, nn.BatchNorm2d):
        #         nn.init.constant_(module.weight, 1)
        #         nn.init.constant_(module.bias, 0)



def resnet_multiimage_input(num_layers, pretrained=True, num_input_images=1):
    assert num_layers in [18, 50], "Number of layers must be 18 or 50."
    block_type = models.resnet.BasicBlock if num_layers == 18 else models.resnet.Bottleneck
    model = ResNetMultiImageInput(block_type, [2, 2, 2, 2] if num_layers == 18 else [3, 4, 6, 3], num_input_images=num_input_images)

    if pretrained:
        # state_dict = model_zoo.load_url(models.resnet.model_urls[f"resnet{num_layers}"], progress=True)
        # state_dict['conv1.weight'] = torch.cat([state_dict['conv1.weight']] * num_input_images, 1) / num_input_images
        # model.load_state_dict(state_dict)
        loaded = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)

    return model






class ResnetEncoder(nn.Module):
    def __init__(self, num_layers=18, pretrained=True, num_input_images=1):
        super().__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if not pretrained:
            self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features


