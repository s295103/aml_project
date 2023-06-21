import torch
import torchvision
import torch.nn as nn
from typing import Optional, Callable, Type, Union
from torchvision.models.resnet import Bottleneck, conv3x3, conv1x1

class BasicBlock(torchvision.models.resnet.BasicBlock):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: str=None,
    ) -> None:
        super(torchvision.models.resnet.BasicBlock, self).__init__()
        if norm_layer == "bn":
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.GroupNorm(groups, planes)
            self.bn2 = nn.GroupNorm(groups, planes)
        if base_width != 64:
            raise ValueError("BasicBlock only supports base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(torchvision.models.resnet.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=[1, 2, 2, 2], pool='avg'):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self._norm_layer = norm_layer if norm_layer is not None else "bn"
        self.groups = groups

        self.inplanes = base_width
        if self._norm_layer == "bn":
                self.bn1 = nn.BatchNorm2d(self.inplanes)
        else:
            self.bn1 = nn.GroupNorm(self.groups, self.inplanes)

        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc_in_features = width // 2 * block.expansion
        self.fc_out_features = num_classes
        self.fc = nn.Linear(self.fc_in_features, self.fc_out_features)

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
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion) if self._norm_layer == "bn" else nn.GroupNorm(self.groups, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
        

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def set_output_layer(self, fc_layer:nn.Module):
        self.fc = fc_layer

if __name__ == "__main__":
    from torchsummary import summary
        # This function creates a fully connected layer to stack on top of the feature extractor
    def make_fc_layer(in_feats:int, out_feats:int, hidden_feats:list ) -> nn.Module:
        if len(hidden_feats) == 0:
            return nn.Sequential(
                nn.BatchNorm1d(in_feats),
                nn.Linear(in_feats, out_feats)
            )
        else:
            layers = []
            layers.append(nn.BatchNorm1d(in_feats))
            layers.append(nn.Linear(in_feats, hidden_feats[0]))
            for i in range(0, len(hidden_feats)-1):
                layers.append(nn.BatchNorm1d(hidden_feats[i]))
                layers.append(nn.Linear(hidden_feats[i], hidden_feats[i+1]))
            layers.append(nn.BatchNorm1d(hidden_feats[-1]))
            layers.append(nn.Linear(hidden_feats[-1], out_feats))
            return nn.Sequential(*layers)
    
    net = ResNet(BasicBlock, [3, 3, 3], num_classes=10, groups=2, norm_layer="gn")
    net.set_output_layer(make_fc_layer(net.fc_in_features, 100, [256, 512, 256]))
    summary(net, (3, 32, 32))

