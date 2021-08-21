from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torch.nn.functional import interpolate

from resNet import BasicBlock, Bottleneck

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LadderConnection(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int
                 ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        B, C, H, W = skip.size()
        skip = self.conv1x1(self.relu(self.norm1(skip)))
        out = interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        out = torch.add(out, skip)
        out = self.conv3x3(self.relu(self.norm2(out)))
        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            make_segmentation_head: bool = False,
            interpolate_to_input_dimensions: bool = False,
            input_channels: int = 3
    ) -> None:
        super(ResNet, self).__init__()
        self.make_segmentation_head = make_segmentation_head
        self.interpolate_to_input_dimensions = interpolate_to_input_dimensions
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.ladder_connection1 = LadderConnection(in_channels=64, out_channels=num_classes)
        self.ladder_connection2 = LadderConnection(in_channels=128, out_channels=num_classes)
        self.ladder_connection3 = LadderConnection(in_channels=256, out_channels=num_classes)
        self.ladder_connection4 = LadderConnection(in_channels=512, out_channels=num_classes)

        if self.make_segmentation_head:
            self.outputs = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, bias=False)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        B, C, H, W = x.shape

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        skip_connection1 = x
        x = self.layer2(x)
        skip_connection2 = x
        x = self.layer3(x)
        skip_connection3 = x
        x = self.layer4(x)
        skip_connection4 = x

        if self.make_segmentation_head:
            x = self.outputs(x)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        x = self.ladder_connection4(x, skip_connection4)
        x = self.ladder_connection3(x, skip_connection3)
        x = self.ladder_connection2(x, skip_connection2)
        x = self.ladder_connection1(x, skip_connection1)

        if self.interpolate_to_input_dimensions:
            x = interpolate(x, size=(H, W), mode="bilinear")
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        in_channels: int,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, input_channels=in_channels, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        if in_channels == 1:
            input_conv_weights = state_dict["conv1.weight"]
            state_dict["conv1.weight"] = input_conv_weights.sum(dim=1, keepdim=True)

        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, in_channels,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, in_channels,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, in_channels,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, in_channels,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, in_channels,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, in_channels, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, in_channels, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, in_channels, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, in_channels=3, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels for first convolution
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, in_channels, **kwargs)
