from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from types import FunctionType
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)
try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50"
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth"
}


def _log_api_usage_once(obj: Any) -> None:

    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
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
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
                norm_layer(planes * block.expansion),
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

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1,x2,x3,x4
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        #
        # return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


class PositionAttentionModule(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x
        return out


class Light_PAM(nn.Module):
    def __init__(self, inchannels, P_h, P_w, **kwargs):
        super(Light_PAM, self).__init__()
        self.att1 = PositionAttentionModule(inchannels)
        self.att2 = PositionAttentionModule(inchannels)
        self.inchalles = inchannels
        self.P_h = P_h
        self.P_w = P_w

    def forward(self,x):
        N,C,H,W =x.size()
        Q_h, Q_w = H//self.P_h,W//self.P_w
        x=x.reshape(N,C,Q_h,self.P_h,Q_w,self.P_w)

        x = x.permute(0,3,5,1,2,4)
        x = x.reshape(N*self.P_h*self.P_w, C, Q_h, Q_w)
        x = self.att1(x)
        x = x.reshape(N,self.P_h, self.P_w, C, Q_h,  Q_w)

        x = x.permute(0, 4,5,3,1,2)
        x = x.reshape(N *Q_h* Q_w, C,self.P_h, self.P_w)
        x = self.att2(x)
        x = x.reshape(N, Q_h, Q_w, C, self.P_h, self.P_w)
        return x.permute(0,3,1,4,2,5).reshape(N,C,H,W)


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""
    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x
        return out


class PosAttDiff(nn.Module):
    """ Difference based on the Position attention module """
    def __init__(self, in_channels, **kwargs):
        super(PosAttDiff, self).__init__()
        self.catt = ChannelAttentionModule()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        batch_size, _, height, width = x1.size()
        # x1 = self.catt(x1)
        # x2 = self.catt(x2)
        feat_a = self.conv_b(x1).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_b = self.conv_c(x2).view(batch_size, -1, height * width)
        feat_1 = self.conv_d(x1).view(batch_size, -1, height * width)
        attention_12 = self.softmax(torch.bmm(feat_a, feat_b))

        feat_c = self.conv_b(x2).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_d = self.conv_c(x1).view(batch_size, -1, height * width)
        feat_2 = self.conv_d(x2).view(batch_size, -1, height * width)
        attention_21 = self.softmax(torch.bmm(feat_c, feat_d))

        attention = torch.abs(attention_12-attention_21)

        feat_1 = torch.bmm(feat_1, attention.permute(0, 2, 1)).view(batch_size, -1, height, width)
        feat_2 = torch.bmm(feat_2, attention.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_1 + self.beta * feat_2
        return out

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.conv = nn.Conv2d(512, 512, 1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        # self.cam5 = ChannelAttentionModule()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    # def forward(self, x):
    #     dilate1_out = nonlinearity(self.dilate1(x))
    #     dilate2_out = nonlinearity(self.dilate2(dilate1_out))
    #     dilate3_out = nonlinearity(self.dilate3(dilate2_out))
    #     dilate4_out = nonlinearity(self.dilate4(dilate3_out))
    #     # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
    #     out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
    #     return out
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(x))
        dilate3_out = nonlinearity(self.dilate3(x))
        dilate4_out = nonlinearity(self.dilate4(x))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        # out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        out = (x + dilate1_out + dilate2_out + dilate3_out + dilate4_out)/5.0  # + dilate5_out
        # out = torch.cat((x, dilate1_out, dilate2_out, dilate3_out, dilate4_out), dim=1)
        # out = self.cam5(out)
        out = self.conv(out)
        out = self.bn(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class ESEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False):
        super(ESEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x.mean((2,3),keepdim=True)
        if self.add_maxpool:
            module_input = 0.5*module_input+0.5*x.amax((2,3),keepdim=True)
        module_input = self.fc(module_input)
        return x * self.sigmoid(module_input)



class BitPosAttMD(nn.Module):
    """ different based on Position attention module"""
    def __init__(self, in_channels, sampler=4, **kwargs):
        super(BitPosAttMD, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.catt = SEModule(in_channels,reduction=4)
        self.conv_b = nn.Conv2d(in_channels, in_channels // sampler, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // sampler, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.segmoid = nn.Sigmoid()

    def forward(self, x1,x2,x):
        batch_size, _, height, width = x1.size()
        x1 = self.bn(x1)
        x2 = self.bn(x2)

        feat_a = self.conv_b(x1).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_b = self.conv_c(x2).view(batch_size, -1, height * width)
        feat_1 = self.conv_d(x).view(batch_size, -1, height * width)

        attention_12 = self.segmoid(self.softmax(torch.bmm(feat_a, feat_b)))
        attention = 1 - attention_12
        feat = torch.bmm(feat_1, attention.permute(0, 2, 1)).view(batch_size, -1, height, width)
        return self.alpha * feat + x

class UpFusion(nn.Module):
    """ different based on Position attention module"""
    def __init__(self, **kwargs):
        super(UpFusion, self).__init__()

        self.conv_4 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv_3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)


        self.conv_lab = nn.Conv2d(1, 1,kernel_size=3,stride=2,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x1,x2,x3,x4, old):
        # batch_size, _, height, width = x1.size()
        out4 = self.conv_4(x4)
        out4 = self.bn4(out4)
        # out3 = self.conv_3(torch.cat([x3, out4],dim=1))
        out3 = self.conv_3(x3+ out4)
        out3 = self.bn3(out3)
        # out2 = self.conv_2(torch.cat([x2, out3],dim=1))
        out2 = self.conv_2(x2+ out3)
        out2 = self.bn2(out2)
        # out1 = self.conv_1(torch.cat([x1, out2],dim=1))
        out1 = self.conv_1(x1+ out2)
        out1 = self.bn1(out1)
        # out1 = self.relu(out1)

        lab = self.conv_lab(torch.unsqueeze(old,dim=1))
        out_new = out1 * (1 - lab)
        return out_new

class New_Fusion(nn.Module):
    """ different based on Position attention module"""
    def __init__(self, **kwargs):
        super(New_Fusion, self).__init__()

        self.deconv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        # self.drl1 = nn.ReLU()
        self.deconv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        # self.drl2 = nn.ReLU()
        self.deconv3 = nn.Conv2d(384, 192, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(192)
        # self.drl3 = nn.ReLU()
        self.deconv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv_lab = nn.Conv2d(1, 1,kernel_size=3,stride=2,padding=1)
        self.relu = nn.ReLU()
        # self.sel = SEModule(128, reduction=2)

    def forward(self, x1,x2,x3,x4, old):

        d1 = self.deconv1(x4)
        d1 = self.norm1(d1)
        d1 = F.interpolate(d1, x3.size()[2:], mode='bilinear', align_corners=False)
        d2 = self.deconv2(torch.cat([d1, x3], dim=1))
        d2 = self.norm2(d2)
        d2 = F.interpolate(d2, x2.size()[2:], mode='bilinear', align_corners=False)
        d3 = self.deconv3(torch.cat([d2, x2], dim=1))
        d3 = self.norm3(d3)
        d3 = F.interpolate(d3, x1.size()[2:], mode='bilinear', align_corners=False)
        out = self.deconv4(torch.cat([d3, x1], dim=1))
        # out = self.sel(outf)
        outf = self.norm4(out)
        out = F.interpolate(outf, (2*x1.size()[2],2*x1.size()[3]), mode='bilinear', align_corners=False)

        lab = self.conv_lab(torch.unsqueeze(old,dim=1))
        backfeat = out * (1 - lab)
        return backfeat, outf

# from DCNv2_latest.dcn_v2 import DCNv2,DCN

# class AttDinkNet34(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3, pretrained=False):
#         super(AttDinkNet34, self).__init__()
#         filters = [64, 128, 256, 512]
#         # filters = [128, 256, 512, 1024]
#         self.resnet_features = resnet34(pretrained=True)
#         self.dcn1 = DCN(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         self.dcn2 = DCN(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         self.dcn3 = DCN(512, 512, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         self.dcn4 = DCN(1024, 1024, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.resx = BasicBlock(64, 64, stride=1, downsample=None)
#
#         # self.dcn41 = DCN(2048, 2048, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn42 = DCN(2048, 2048, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#
#         # self.conv_offset12 = nn.Conv2d(512, 2 * 2 * 3 * 3, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         # self.conv_mask12 = nn.Conv2d(512, 2 * 1 * 3 * 3,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1), bias=True).cuda()
#         # self.conv_offset12.weight.data.zero_()
#         # self.conv_offset12.bias.data.zero_()
#         # self.conv_mask12.weight.data.zero_()
#         # self.conv_mask12.bias.data.zero_()
#         #
#         # self.conv_offset22 = nn.Conv2d(512, 2 * 2 * 3 * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True).cuda()
#         # self.conv_mask22 = nn.Conv2d(512, 2 * 1 * 3 * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True).cuda()
#         # self.conv_offset22.weight.data.zero_()
#         # self.conv_offset22.bias.data.zero_()
#         # self.conv_mask22.weight.data.zero_()
#         # self.conv_mask22.bias.data.zero_()
#         self.att4 = BitPosAttMD(512, 8)
#         self.att3 = BitPosAttMD(256, 4)
#         self.att2 = BitPosAttMD(128, 2)
#         self.att1 = BitPosAttMD(64, 1)
#         self.conv4 = nn.Conv2d(1024, 512, 3, padding=1, bias=False)
#         # self.c4 = Diff_Block(2048,fusion='all')
#         # self.bn4 = nn.BatchNorm2d(1024)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.rl4 = nn.ReLU()
#
#         self.conv3 = nn.Conv2d(512, 256, 3, padding=1, bias=False)
#         # self.c3 = Diff_Block(1024,fusion='all')
#         # self.bn3 = nn.BatchNorm2d(512)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.rl3 = nn.ReLU()
#         self.conv_offset1 = nn.Conv2d(128, 36,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         self.conv_mask1 = nn.Conv2d(128, 18,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         self.dcn21 = DCNv2(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv_offset2 = nn.Conv2d(256, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.conv_mask2 = nn.Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.dcn22 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv_offset3 = nn.Conv2d(512, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.conv_mask3 = nn.Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.dcn23 = DCNv2(512, 512, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv_offset4 = nn.Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.conv_mask4 = nn.Conv2d(1024, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.dcn24 = DCNv2(1024, 1024, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
#         # self.nbk2 = NBlock(256)
#         # self.c2 = Diff_Block(512,fusion='all')
#         # self.bn2 = nn.BatchNorm2d(256)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.rl2 = nn.ReLU()
#         # self.dcn11 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         # self.dcn12 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         self.conv1 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
#         # self.nbk1 = NBlock(128)
#         # self.c1 = Diff_Block(256,fusion='all')
#         # self.bn1 = LayerNorm(128)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.rl1 = nn.ReLU()
#
#         self.attdiff4 = PosAttDiff(2048)
#         self.attdiff3 = PosAttDiff(1024)
#         self.attdiff2 = PosAttDiff(512)
#         self.attdiff1 = PosAttDiff(256)
#         self.dblock = Dblock(512)
#         self.pam1 = PositionAttentionModule(256)
#         self.pam2 = PositionAttentionModule(512)
#         self.pam3 = PositionAttentionModule(1024)
#         self.pam4 = PositionAttentionModule(1024)
#         # self.cam1 = SEModule(channels=1024,reduction=4)
#         # self.cam2 = SEModule(channels=128,reduction=1)
#         # self.sel = SEModule(channels=128,reduction=1)
#
#         self.down1 = nn.Conv2d(1024, 128, kernel_size=1, padding=0, bias=False)
#         self.down2 = nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False)
#         self.down3 = nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)
#         self.down4 = nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=False)
#
#         self.res1 = BasicBlock(128, 128)
#         self.res2 = BasicBlock(256, 256)
#         self.res3 = BasicBlock(512, 512)
#
#         self.fuse = nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=False)
#         self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
#
#         self.fuse3 = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)
#         self.fuse2 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 64, 4, 2, 1)
#         self.finalrelu1 = nn.ReLU()
#         self.finalconv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.finalrelu2 = nn.ReLU()
#         self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1)
#
#         self.seg11 = nn.Conv2d(256, 32, 3, padding=1)
#         self.seg12 = nn.Conv2d(256, 32, 3, padding=1)
#         self.bn11 = nn.BatchNorm2d(128)
#         self.seg21 = nn.Conv2d(512, 32, 3, padding=1)
#         self.seg22 = nn.Conv2d(512, 32, 3, padding=1)
#         self.bn22 = nn.BatchNorm2d(128)
#         self.seg31 = nn.Conv2d(1024, 64, 3, padding=1)
#         self.seg32 = nn.Conv2d(1024, 64, 3, padding=1)
#         self.bn33 = nn.BatchNorm2d(128)
#         self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
#         self.norm1 = nn.BatchNorm2d(256)
#
#         self.deconv2 = nn.Conv2d(256, 128, 3, padding=1)
#         self.norm2 = nn.BatchNorm2d(128)
#
#         self.deconv3 = nn.Conv2d(128, 64, 3, padding=1)
#         self.norm3 = nn.BatchNorm2d(64)
#
#         self.deconv4 = nn.Conv2d(64, 32, 3, padding=1)
#         self.norm4 = nn.BatchNorm2d(32)
#
#         self.deconv5 = nn.Conv2d(32, 32, 3, padding=1)
#         self.norm5 = nn.BatchNorm2d(32)
#
#         self.finalseg = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, input1,input2):
#         # Encoder
#         e11, e12, e13, e14 = self.resnet_features(input1)
#         e21, e22, e23, e24 = self.resnet_features(input2)
#         x_size = input1.size()
#
#         offset1 = self.conv_offset1(torch.cat((e11, e21),dim=1))
#         mask1 = self.conv_mask1(torch.cat((e11, e21),dim=1))
#         # mask1 = torch.sigmoid(mask1)
#         e1 = self.dcn21(torch.cat((e11, e21),dim=1), offset1, mask1)
#
#         # e1 = self.dcn1(torch.cat((e11, e21),dim=1))
#         e1 = self.conv1(e1)
#         # e1 = self.conv1(torch.abs(e11 - e21))  # 128 128 128
#         e1 = self.bn1(e1)
#         e1 = self.rl1(e1)
#
#         offset2 = self.conv_offset2(torch.cat((e12, e22), dim=1))
#         mask2 = self.conv_mask2(torch.cat((e12, e22), dim=1))
#         # mask2 = torch.sigmoid(mask2)
#         e2 = self.dcn22(torch.cat((e12, e22), dim=1),offset2, mask2)
#         e2 = self.conv2(e2)
#         # e2 = self.conv2(torch.abs(e12 - e22))  # 256 64 64
#         e2 = self.bn2(e2)
#         e2 = self.rl2(e2)
#
#         offset3 = self.conv_offset3(torch.cat((e13, e23), dim=1))
#         mask3 = self.conv_mask3(torch.cat((e13, e23), dim=1))
#         # mask3 = torch.sigmoid(mask3)
#         e3 = self.dcn23(torch.cat((e13, e23), dim=1),offset3, mask3)
#         e3 = self.conv3(e3)
#         # e3 = self.conv3(torch.abs(e13 - e23))  # 512 32 32
#         e3 = self.bn3(e3)
#         e3 = self.rl3(e3)
#
#         offset4 = self.conv_offset4(torch.cat((e14, e24), dim=1))
#         mask4 = self.conv_mask4(torch.cat((e14, e24), dim=1))
#         # mask4 = torch.sigmoid(mask4)
#         e4 = self.dcn24(torch.cat((e14, e24), dim=1),offset4, mask4)
#         e4 = self.conv4(e4)
#         # e4 = self.conv4(torch.abs(e14 - e24))  # 1024 16 16
#         e4 = self.bn4(e4)
#         e4 = self.rl4(e4)
#
#         # e4 = self.dblock(e4)
#
#         # d4 = self.decoder4(e4) + e3
#         # d3 = self.decoder3(d4) + e2
#         # d2 = self.decoder2(d3) + e1
#         # d1 = self.decoder1(d2)
#         #
#         # out = self.finaldeconv1(d1)
#         # out = self.finalrelu1(out)
#         # out = self.finalconv2(out)
#         # out = self.finalrelu2(out)
#         # out = self.finalconv3(out)
#
#         d1 = self.deconv1(e4)
#         d1 = self.norm1(d1)
#         d1 = F.interpolate(d1, e3.size()[2:],mode='bilinear')
#
#         d2 = self.deconv2(d1+e3)
#         d2 = self.norm2(d2)
#         d2 = F.interpolate(d2,  e2.size()[2:], mode='bilinear')
#
#         d3 = self.deconv3(d2+e2)
#         d3 = self.norm3(d3)
#         d3 = F.interpolate(d3, e1.size()[2:], mode='bilinear')
#
#         d4 = self.deconv4(d3+e1)
#         d4 = self.norm4(d4)
#         d4 = F.interpolate(d4, x_size[2:], mode='bilinear')
#
#         out = self.finalseg(d4)
#
#         return out

# class BaselineMod(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3, pretrained=False):
#         super(BaselineMod, self).__init__()
#         filters = [64, 128, 256, 512]
#         # filters = [128, 256, 512, 1024]
#         self.resnet_features = resnet34(pretrained=True)
#         # self.dcn1 = DCN(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn2 = DCN(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn3 = DCN(512, 512, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn4 = DCN(1024, 1024, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.resx = BasicBlock(64, 64, stride=1, downsample=None)
#
#         # self.dcn41 = DCN(2048, 2048, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn42 = DCN(2048, 2048, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#
#         # self.conv_offset12 = nn.Conv2d(512, 2 * 2 * 3 * 3, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         # self.conv_mask12 = nn.Conv2d(512, 2 * 1 * 3 * 3,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1), bias=True).cuda()
#         # self.conv_offset12.weight.data.zero_()
#         # self.conv_offset12.bias.data.zero_()
#         # self.conv_mask12.weight.data.zero_()
#         # self.conv_mask12.bias.data.zero_()
#         #
#         # self.conv_offset22 = nn.Conv2d(512, 2 * 2 * 3 * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True).cuda()
#         # self.conv_mask22 = nn.Conv2d(512, 2 * 1 * 3 * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True).cuda()
#         # self.conv_offset22.weight.data.zero_()
#         # self.conv_offset22.bias.data.zero_()
#         # self.conv_mask22.weight.data.zero_()
#         # self.conv_mask22.bias.data.zero_()
#         # self.att4 = BitPosAttMD(512, 8)
#         # self.att3 = BitPosAttMD(256, 4)
#         # self.att2 = BitPosAttMD(128, 2)
#         # self.att1 = BitPosAttMD(64, 1)
#         self.conv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#         # self.c4 = Diff_Block(2048,fusion='all')
#         # self.bn4 = nn.BatchNorm2d(1024)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.rl4 = nn.ReLU()
#
#         self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#         # self.c3 = Diff_Block(1024,fusion='all')
#         # self.bn3 = nn.BatchNorm2d(512)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.rl3 = nn.ReLU()
#         # self.conv_offset1 = nn.Conv2d(128, 36,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         # self.conv_mask1 = nn.Conv2d(128, 18,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         # self.dcn21 = DCNv2(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         #
#         # self.conv_offset2 = nn.Conv2d(256, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.conv_mask2 = nn.Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.dcn22 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         #
#         # self.conv_offset3 = nn.Conv2d(512, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.conv_mask3 = nn.Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.dcn23 = DCNv2(512, 512, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         #
#         # self.conv_offset4 = nn.Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.conv_mask4 = nn.Conv2d(1024, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.dcn24 = DCNv2(1024, 1024, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
#         # self.nbk2 = NBlock(256)
#         # self.c2 = Diff_Block(512,fusion='all')
#         # self.bn2 = nn.BatchNorm2d(256)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.rl2 = nn.ReLU()
#         # self.dcn11 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         # self.dcn12 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
#         # self.nbk1 = NBlock(128)
#         # self.c1 = Diff_Block(256,fusion='all')
#         # self.bn1 = LayerNorm(128)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.rl1 = nn.ReLU()
#
#         # self.attdiff4 = PosAttDiff(2048)
#         # self.attdiff3 = PosAttDiff(1024)
#         # self.attdiff2 = PosAttDiff(512)
#         # self.attdiff1 = PosAttDiff(256)
#         # self.dblock = Dblock(512)
#         # self.pam1_1 = PositionAttentionModule(64)
#         # self.pam1_2 = PositionAttentionModule(64)
#         # self.pam2 = PositionAttentionModule(512)
#         # self.pam3 = PositionAttentionModule(1024)
#         # self.pam4 = PositionAttentionModule(1024)
#         # self.cam1 = SEModule(channels=1024,reduction=4)
#         # self.cam2 = SEModule(channels=128,reduction=1)
#         # self.sel11 = SEModule(channels=64,reduction=1)
#         # self.sel12 = SEModule(channels=64, reduction=1)
#
#         # self.down1 = nn.Conv2d(1024, 128, kernel_size=1, padding=0, bias=False)
#         # self.down2 = nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False)
#         # self.down3 = nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)
#         # self.down4 = nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=False)
#         #
#         # self.res1 = BasicBlock(128, 128)
#         # self.res2 = BasicBlock(256, 256)
#         # self.res3 = BasicBlock(512, 512)
#         #
#         # self.fuse = nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=False)
#         # self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
#         #
#         # self.fuse3 = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)
#         # self.fuse2 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)
#
#         # self.decoder4 = DecoderBlock(filters[3], filters[2])
#         # self.decoder3 = DecoderBlock(filters[2], filters[1])
#         # self.decoder2 = DecoderBlock(filters[1], filters[0])
#         # self.decoder1 = DecoderBlock(filters[0], filters[0])
#         #
#         # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 64, 4, 2, 1)
#         # self.finalrelu1 = nn.ReLU()
#         # self.finalconv2 = nn.Conv2d(64, 64, 3, padding=1)
#         # self.finalrelu2 = nn.ReLU()
#         # self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1)
#
#         # self.seg11 = nn.Conv2d(256, 32, 3, padding=1)
#         # self.seg12 = nn.Conv2d(256, 32, 3, padding=1)
#         # self.bn11 = nn.BatchNorm2d(128)
#         # self.seg21 = nn.Conv2d(512, 32, 3, padding=1)
#         # self.seg22 = nn.Conv2d(512, 32, 3, padding=1)
#         # self.bn22 = nn.BatchNorm2d(128)
#         # self.seg31 = nn.Conv2d(1024, 64, 3, padding=1)
#         # self.seg32 = nn.Conv2d(1024, 64, 3, padding=1)
#         # self.bn33 = nn.BatchNorm2d(128)
#
#         self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
#         self.norm1 = nn.BatchNorm2d(256)
#
#         self.deconv2 = nn.Conv2d(256, 128, 3, padding=1)
#         self.norm2 = nn.BatchNorm2d(128)
#
#         self.deconv3 = nn.Conv2d(128, 64, 3, padding=1)
#         self.norm3 = nn.BatchNorm2d(64)
#
#         self.deconv4 = nn.Conv2d(64, 32, 3, padding=1)
#         self.norm4 = nn.BatchNorm2d(32)
#
#         # self.deconv5 = nn.Conv2d(32, 32, 3, padding=1)
#         # self.norm5 = nn.BatchNorm2d(32)
#
#         self.finalseg = nn.Conv2d(32, num_classes, 3, padding=1)
#
#         # self.sbn1 = nn.BatchNorm2d(960)
#         self.convblock = nn.Conv2d(960, 64, 3, padding=1)
#         self.sbn2 = nn.BatchNorm2d(64)
#         self.segblock = nn.Conv2d(64, num_classes, 3, padding=1)
#
#     def forward(self, input1,input2):
#         # Encoder
#         e11, e12, e13, e14 = self.resnet_features(input1)
#         e21, e22, e23, e24 = self.resnet_features(input2)
#         x_size = input1.size()
#
#         # offset1 = self.conv_offset1(torch.cat((e11, e21),dim=1))
#         # mask1 = self.conv_mask1(torch.cat((e11, e21),dim=1))
#         # mask1 = torch.sigmoid(mask1)
#         # e1 = self.dcn21(torch.cat((e11, e21),dim=1), offset1, mask1)
#
#         # e1 = self.dcn1(torch.cat((e11, e21),dim=1))
#         # e1 = self.conv1(e1)
#         e1 = self.conv1(torch.abs(e11 - e21))  # 128 128 128
#         e1 = self.bn1(e1)
#         e1 = self.rl1(e1)
#
#         # offset2 = self.conv_offset2(torch.cat((e12, e22), dim=1))
#         # mask2 = self.conv_mask2(torch.cat((e12, e22), dim=1))
#         # mask2 = torch.sigmoid(mask2)
#         # e2 = self.dcn22(torch.cat((e12, e22), dim=1),offset2, mask2)
#         # e2 = self.conv2(e2)
#         e2 = self.conv2(torch.abs(e12 - e22))  # 256 64 64
#         e2 = self.bn2(e2)
#         e2 = self.rl2(e2)
#
#         # offset3 = self.conv_offset3(torch.cat((e13, e23), dim=1))
#         # mask3 = self.conv_mask3(torch.cat((e13, e23), dim=1))
#         # mask3 = torch.sigmoid(mask3)
#         # e3 = self.dcn23(torch.cat((e13, e23), dim=1),offset3, mask3)
#         # e3 = self.conv3(e3)
#         e3 = self.conv3(torch.abs(e13 - e23))  # 512 32 32
#         e3 = self.bn3(e3)
#         e3 = self.rl3(e3)
#
#         # offset4 = self.conv_offset4(torch.cat((e14, e24), dim=1))
#         # mask4 = self.conv_mask4(torch.cat((e14, e24), dim=1))
#         # mask4 = torch.sigmoid(mask4)
#         # e4 = self.dcn24(torch.cat((e14, e24), dim=1),offset4, mask4)
#         # e4 = self.conv4(e4)
#         e4 = self.conv4(torch.abs(e14 - e24))  # 1024 16 16
#         e4 = self.bn4(e4)
#         e4 = self.rl4(e4)
#
#         # e4 = self.dblock(e4)
#
#         se12 = F.interpolate(e12, e11.size()[2:], mode='bilinear',align_corners=False)
#         se13 = F.interpolate(e13, e11.size()[2:], mode='bilinear',align_corners=False)
#         se14 = F.interpolate(e14, e11.size()[2:], mode='bilinear',align_corners=False)
#
#         se22 = F.interpolate(e22, e21.size()[2:], mode='bilinear',align_corners=False)
#         se23 = F.interpolate(e23, e21.size()[2:], mode='bilinear',align_corners=False)
#         se24 = F.interpolate(e24, e21.size()[2:], mode='bilinear',align_corners=False)
#
#         seg1 = self.convblock(torch.cat((e11,se12,se13,se14),dim=1))
#         # seg1 = self.sel11(seg1)
#         seg1 = self.sbn2(seg1)
#         seg1 = F.interpolate(seg1, x_size[2:], mode='bilinear',align_corners=False)
#         seg1 = self.segblock(seg1)
#
#         seg2 = self.convblock(torch.cat((e21, se22, se23, se24), dim=1))
#         # seg2 = self.sel12(seg2)
#         seg2 = self.sbn2(seg2)
#         seg2 = F.interpolate(seg2, x_size[2:], mode='bilinear',align_corners=False)
#         seg2 = self.segblock(seg2)
#
#
#
#         d1 = self.deconv1(e4)
#         d1 = self.norm1(d1)
#         d1 = F.interpolate(d1, e3.size()[2:],mode='bilinear',align_corners=False)
#
#         d2 = self.deconv2(d1+e3)
#         d2 = self.norm2(d2)
#         d2 = F.interpolate(d2,  e2.size()[2:], mode='bilinear',align_corners=False)
#
#         d3 = self.deconv3(d2+e2)
#         d3 = self.norm3(d3)
#         d3 = F.interpolate(d3, e1.size()[2:], mode='bilinear',align_corners=False)
#
#         d4 = self.deconv4(d3+e1)
#         d4 = self.norm4(d4)
#         d4 = F.interpolate(d4, x_size[2:], mode='bilinear',align_corners=False)
#
#         out = self.finalseg(d4)
#
#         return out,seg1,seg2
#         # return out


# class Baseline34(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3, pretrained=False):
#         super(Baseline34, self).__init__()
#         self.resnet_features = resnet34(pretrained=True)
#
#         self.conv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.rl4 = nn.ReLU()
#         # self.lpam4 = Light_PAM(512, 2, 2)
#         #
#         self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.rl3 = nn.ReLU()
#         # self.lpam3 = Light_PAM(256, 4, 4)
#         #
#         self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.rl2 = nn.ReLU()
#         # self.lpam2 = Light_PAM(128, 8, 8)
#         #
#         self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.rl1 = nn.ReLU()
#         # self.lpam1 = Light_PAM(64,16,16)
#
#         # # self.dpam1 = PositionAttentionModule(64)
#         # self.deca1 = ChannelAttentionModule()
#         #
#         # self.dpam2 = PositionAttentionModule(128)
#         # self.deca2 = ChannelAttentionModule()
#         #
#         # self.dpam3 = PositionAttentionModule(256)
#         # self.deca3 = ChannelAttentionModule()
#         #
#         # self.dpam4 = PositionAttentionModule(512)
#         # self.deca4 = ChannelAttentionModule()
#
#         # self.dblock = Dblock(512)
#         #
#         # self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
#         # self.norm1 = nn.BatchNorm2d(256)
#         # self.drl1 = nn.ReLU()
#         # self.deconv2 = nn.Conv2d(256, 128, 3, padding=1)
#         # self.norm2 = nn.BatchNorm2d(128)
#         # self.drl2 = nn.ReLU()
#         # self.deconv3 = nn.Conv2d(128, 64, 3, padding=1)
#         # self.norm3 = nn.BatchNorm2d(64)
#         # self.drl3 = nn.ReLU()
#         # self.deconv4 = nn.Conv2d(64, 32, 3, padding=1)
#         # self.norm4 = nn.BatchNorm2d(32)
#         # self.drl4 = nn.ReLU()
#         # self.finalseg = nn.Conv2d(32, num_classes, 3, padding=1)
#
#         filters = [64, 128, 256, 512]
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], 64)
#         # self.decoder0 = DecoderBlock(32, num_classes)
#         self.finaldeconv1 = nn.Conv2d(64, 32, 3, padding=1)
#         self.finalrelu1 = nn.BatchNorm2d(32)
#         self.finalconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
#         self.finalrelu2 = nn.BatchNorm2d(32)
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#
#     def forward(self, inputs):
#         # Encoder
#         e11, e12, e13, e14 = self.resnet_features(inputs)
#         x_size = inputs.size()
#
#         # e1 = self.deca1(e11)
#         # e1 = self.bn1(e1)
#         # e2 = self.deca2(self.dpam2(e12))
#         # e2 = self.bn2(e2)
#         # e3 = self.deca3(self.dpam3(e13))
#         # e3 = self.bn3(e3)
#         # e4 = self.deca4(self.dpam4(e14))
#         # e4 = self.bn4(e4)
#         e1 = self.conv1(e11)  # 128 128 128
#         e1 = self.bn1(e1)
#         e1 = self.rl1(e1)
#         # e1 = self.lpam1(e1)
#
#         e2 = self.conv2(e12)  # 256 64 64
#         e2 = self.bn2(e2)
#         e2 = self.rl2(e2)
#         # e2 = self.lpam2(e2)
#
#         e3 = self.conv3(e13)  # 512 32 32
#         e3 = self.bn3(e3)
#         e3 = self.rl3(e3)
#         # e3 = self.lpam3(e3)
#
#         e4 = self.conv4(e14)  # 1024 16 16
#         e4 = self.bn4(e4)
#         e4 = self.rl4(e4)
#         # e4 = self.lpam4(e4)
#         # e4 = self.dblock(e4)
#
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         # out = self.decoder0(d1)
#
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return out


from DCNv2.dcn_v2 import DCN


class Baseline34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=False):
        super(Baseline34, self).__init__()
        filters = [64, 128, 256, 512]
        self.resnet_features = resnet34(pretrained=True)

        self.conv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        # self.rl4 = nn.ReLU()

        self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        # self.rl3 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        # self.rl2 = nn.ReLU()

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.rl1 = nn.ReLU()

        # self.dblock = Dblock(512)
        # self.fusin = UpFusion()
        self.fusin = New_Fusion()
        # self.pam2 = PositionAttentionModule(256)
        # self.pam3 = PositionAttentionModule(512)

        self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        # self.drl1 = nn.ReLU()
        self.deconv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        # self.drl2 = nn.ReLU()
        self.deconv3 = nn.Conv2d(384, 192, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(192)
        # self.drl3 = nn.ReLU()
        self.deconv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.sel = SEModule(128,reduction=2)
        # self.drl4 = nn.ReLU()
        self.finalseg = nn.Conv2d(128, num_classes, 3, padding=1)

        # self.finalnew = nn.Conv2d(32, num_classes, 3, padding=1)
        self.finalmov = nn.Conv2d(128, num_classes, 3, padding=1)

        # self.conv_lab11 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        # self.bnorm21 = nn.BatchNorm2d(32)
        # self.conv_lab21 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.segblock = nn.Conv2d(128, num_classes, 3, padding=1)
        self.conv_lab1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv_lab2 = nn.Conv2d(64, num_classes, 3, padding=1)

        self.dcn = DCN(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)

    def forward(self, inputs, labelso):
        # Encoder
        # labels_o{1,3},labels_n{2},labels_m{3},labels{1,2}
        e11, e12, e13, e14 = self.resnet_features(inputs)
        x_size = inputs.size()

        e1 = self.conv1(e11)
        e1 = self.bn1(e1)
        # e1 = self.rl1(e1)
        e2 = self.conv2(e12)
        e2 = self.bn2(e2)
        # e2 = self.rl2(e2)
        e3 = self.conv3(e13)
        e3 = self.bn3(e3)
        # e3 = self.pam2(e3)
        # e3 = self.rl3(e3)
        e4 = self.conv4(e14)
        e4 = self.bn4(e4)
        # e4 = self.pam3(e4)
        # e4 = self.rl4(e4)
        # e4 = self.dblock(e4)
        fu_new, featn = self.fusin(e1, e2, e3, e4, labelso)

        d1 = self.deconv1(e4)
        d1 = self.norm1(d1)
        d1 = F.interpolate(d1, e3.size()[2:], mode='bilinear', align_corners=False)

        d2 = self.deconv2(torch.cat([d1, e3], dim=1))
        d2 = self.norm2(d2)
        d2 = F.interpolate(d2, e2.size()[2:], mode='bilinear', align_corners=False)

        d3 = self.deconv3(torch.cat([d2, e2], dim=1))
        d3 = self.norm3(d3)
        d3 = F.interpolate(d3, e1.size()[2:], mode='bilinear', align_corners=False)

        d4f1 = self.deconv4(torch.cat([d3, e1], dim=1))
        # d4f = self.norm4(d4f1)
        d4f = self.sel(d4f1)

        d4f = self.dcn(d4f)
        d4 = self.norm4(d4f)
        d4 = F.interpolate(d4, x_size[2:], mode='bilinear', align_corners=False)
        out = self.finalseg(d4)

        # new_out = self.finalnew(d4 *(1-torch.unsqueeze(labelso,dim=1)))
        mov_out = self.finalmov((1 - torch.sigmoid(d4)) * torch.unsqueeze(labelso, dim=1))
        # # mov_out = (1 - torch.sigmoid(out)) * torch.unsqueeze(labelso, dim=1)
        # mov_out = self.conv_lab1(fu_mov)
        # mov_out = self.bnorm2(mov_out)
        # mov_out = self.conv_lab2(mov_out)

        fu_new = self.sel(fu_new)
        fu_new = self.dcn(fu_new)
        # fu_new = self.bnorm1(fu_new)
        new_out = self.conv_lab1(fu_new)
        new_out = self.bnorm2(new_out)
        new_out = self.conv_lab2(new_out)
        feat_all = F.interpolate(self.segblock(d4f), (inputs.size()[2], inputs.size()[3]), mode='bilinear',
                                 align_corners=False)
        feat_mov = F.interpolate(self.segblock(featn), (inputs.size()[2], inputs.size()[3]), mode='bilinear',
                                 align_corners=False)
        # feat_all = self.segblock(d4f)
        # feat_mov = self.segblock(featn)
        # out_new = self.finalseg2(torch.cat((d4,F.interpolate(fusion, inputs.size()[2:], mode='bilinear',align_corners=False)),dim=1))
        return out, mov_out, new_out, feat_all, feat_mov
    # def forward(self, inputs,labelso):
    #     # Encoder
    #     # labels_o{1,3},labels_n{2},labels_m{3},labels{1,2}
    #     e11, e12, e13, e14 = self.resnet_features(inputs)
    #     x_size = inputs.size()
    #
    #     e1 = self.conv1(e11)
    #     e1 = self.bn1(e1)
    #     # e1 = self.rl1(e1)
    #     e2 = self.conv2(e12)
    #     e2 = self.bn2(e2)
    #     # e2 = self.rl2(e2)
    #     e3 = self.conv3(e13)
    #     e3 = self.bn3(e3)
    #     # e3 = self.pam2(e3)
    #     # e3 = self.rl3(e3)
    #     e4 = self.conv4(e14)
    #     e4 = self.bn4(e4)
    #     # e4 = self.pam3(e4)
    #     # e4 = self.rl4(e4)
    #     # e4 = self.dblock(e4)
    #     fu_new,featn = self.fusin(e1,e2,e3,e4,labelso)
    #
    #     d1 = self.deconv1(e4)
    #     d1 = self.norm1(d1)
    #     d1 = F.interpolate(d1, e3.size()[2:],mode='bilinear',align_corners=False)
    #
    #     d2 = self.deconv2(torch.cat([d1,e3],dim=1))
    #     d2 = self.norm2(d2)
    #     d2 = F.interpolate(d2,  e2.size()[2:], mode='bilinear',align_corners=False)
    #
    #     d3 = self.deconv3(torch.cat([d2,e2],dim=1))
    #     d3 = self.norm3(d3)
    #     d3 = F.interpolate(d3, e1.size()[2:], mode='bilinear',align_corners=False)
    #
    #     d4f = self.deconv4(torch.cat([d3, e1],dim=1))
    #     # dcn0 = d4f
    #     d4f = self.dcn(d4f)
    #     # dcn1 = d4f
    #     # d4 = self.sel(d4f)
    #     d4 = self.norm4(d4f)
    #
    #     d4 = F.interpolate(d4, x_size[2:], mode='bilinear',align_corners=False)
    #     # dcn1 = d4
    #     out = self.finalseg(d4)
    #
    #     # new_out = self.finalnew(d4 *(1-torch.unsqueeze(labelso,dim=1)))
    #
    #     mov_out = self.finalmov((1 - torch.sigmoid(d4)) * torch.unsqueeze(labelso, dim=1))
    #     # dcn1 = (1 - torch.sigmoid(d4)) * torch.unsqueeze(labelso, dim=1)
    #     # # mov_out = (1 - torch.sigmoid(out)) * torch.unsqueeze(labelso, dim=1)
    #     # mov_out = self.conv_lab1(fu_mov)
    #     # mov_out = self.bnorm2(mov_out)
    #     # mov_out = self.conv_lab2(mov_out)
    #     # dcn0 = fu_new
    #     fu_new = self.dcn(fu_new)
    #     # dcn1 = fu_new
    #     fu_new = self.sel(fu_new)
    #     new_out = self.conv_lab1(fu_new)
    #     new_out = self.bnorm2(new_out)
    #     new_out = self.conv_lab2(new_out)
    #     feat_all = F.interpolate(self.segblock(d4f), (inputs.size()[2],inputs.size()[3]), mode='bilinear',align_corners=False)
    #     feat_mov = F.interpolate(self.segblock(featn), (inputs.size()[2],inputs.size()[3]), mode='bilinear',align_corners=False)
    #     # feat_all = self.segblock(d4f)
    #     # feat_mov = self.segblock(featn)
    #     # out_new = self.finalseg2(torch.cat((d4,F.interpolate(fusion, inputs.size()[2:], mode='bilinear',align_corners=False)),dim=1))
    #     return out, mov_out, new_out,feat_all, feat_mov

class UpFusionN(nn.Module):
    """ different based on Position attention module"""
    def __init__(self, **kwargs):
        super(UpFusionN, self).__init__()

        self.conv_4 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv_3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)


        self.conv_lab = nn.Conv2d(1, 1,kernel_size=3,stride=2,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x1,x2,x3,x4, old):
        # batch_size, _, height, width = x1.size()
        out4 = self.conv_4(x4)
        out4 = self.bn4(out4)
        out3 = self.conv_3(x3+out4)
        out3 = self.bn3(out3)
        out2 = self.conv_2(x2 + out3)
        out2 = self.bn2(out2)
        out1 = self.conv_1(x1 + out2)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        lab = self.conv_lab(torch.unsqueeze(old,dim=1))
        out_new = out1 * (1 - lab)
        return out_new


class UpFusionM(nn.Module):
    """ different based on Position attention module"""
    def __init__(self, **kwargs):
        super(UpFusionM, self).__init__()

        self.conv_4 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv_3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)


        self.conv_lab = nn.Conv2d(1, 1,kernel_size=3,stride=2,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x1,x2,x3,x4, old):
        # batch_size, _, height, width = x1.size()
        out4 = self.conv_4(x4)
        out4 = self.bn4(out4)
        out3 = self.conv_3(x3+out4)
        out3 = self.bn3(out3)
        out2 = self.conv_2(x2 + out3)
        out2 = self.bn2(out2)
        out1 = self.conv_1(x1 + out2)
        out1 = self.bn1(out1)
        # out1 = self.relu(out1)

        # out_mov = (1 - torch.sigmoid(out1)) * torch.unsqueeze(old, dim=1)
        # return out_mov

        lab = self.conv_lab(torch.unsqueeze(old,dim=1))
        out_mov = (1 - torch.sigmoid(out1)) * lab
        return out_mov

# from DCNv2_latest.dcn_v2 import DCNv2,DCN

# class AttDinkNet34(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3, pretrained=False):
#         super(AttDinkNet34, self).__init__()
#         filters = [64, 128, 256, 512]
#         # filters = [128, 256, 512, 1024]
#         self.resnet_features = resnet34(pretrained=True)
#         self.dcn1 = DCN(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         self.dcn2 = DCN(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         self.dcn3 = DCN(512, 512, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         self.dcn4 = DCN(1024, 1024, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.resx = BasicBlock(64, 64, stride=1, downsample=None)
#
#         # self.dcn41 = DCN(2048, 2048, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn42 = DCN(2048, 2048, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#
#         # self.conv_offset12 = nn.Conv2d(512, 2 * 2 * 3 * 3, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         # self.conv_mask12 = nn.Conv2d(512, 2 * 1 * 3 * 3,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1), bias=True).cuda()
#         # self.conv_offset12.weight.data.zero_()
#         # self.conv_offset12.bias.data.zero_()
#         # self.conv_mask12.weight.data.zero_()
#         # self.conv_mask12.bias.data.zero_()
#         #
#         # self.conv_offset22 = nn.Conv2d(512, 2 * 2 * 3 * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True).cuda()
#         # self.conv_mask22 = nn.Conv2d(512, 2 * 1 * 3 * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True).cuda()
#         # self.conv_offset22.weight.data.zero_()
#         # self.conv_offset22.bias.data.zero_()
#         # self.conv_mask22.weight.data.zero_()
#         # self.conv_mask22.bias.data.zero_()
#         self.att4 = BitPosAttMD(512, 8)
#         self.att3 = BitPosAttMD(256, 4)
#         self.att2 = BitPosAttMD(128, 2)
#         self.att1 = BitPosAttMD(64, 1)
#         self.conv4 = nn.Conv2d(1024, 512, 3, padding=1, bias=False)
#         # self.c4 = Diff_Block(2048,fusion='all')
#         # self.bn4 = nn.BatchNorm2d(1024)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.rl4 = nn.ReLU()
#
#         self.conv3 = nn.Conv2d(512, 256, 3, padding=1, bias=False)
#         # self.c3 = Diff_Block(1024,fusion='all')
#         # self.bn3 = nn.BatchNorm2d(512)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.rl3 = nn.ReLU()
#         self.conv_offset1 = nn.Conv2d(128, 36,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         self.conv_mask1 = nn.Conv2d(128, 18,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         self.dcn21 = DCNv2(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv_offset2 = nn.Conv2d(256, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.conv_mask2 = nn.Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.dcn22 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv_offset3 = nn.Conv2d(512, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.conv_mask3 = nn.Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.dcn23 = DCNv2(512, 512, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv_offset4 = nn.Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.conv_mask4 = nn.Conv2d(1024, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         self.dcn24 = DCNv2(1024, 1024, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
#         # self.nbk2 = NBlock(256)
#         # self.c2 = Diff_Block(512,fusion='all')
#         # self.bn2 = nn.BatchNorm2d(256)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.rl2 = nn.ReLU()
#         # self.dcn11 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         # self.dcn12 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         self.conv1 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
#         # self.nbk1 = NBlock(128)
#         # self.c1 = Diff_Block(256,fusion='all')
#         # self.bn1 = LayerNorm(128)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.rl1 = nn.ReLU()
#
#         self.attdiff4 = PosAttDiff(2048)
#         self.attdiff3 = PosAttDiff(1024)
#         self.attdiff2 = PosAttDiff(512)
#         self.attdiff1 = PosAttDiff(256)
#         self.dblock = Dblock(512)
#         self.pam1 = PositionAttentionModule(256)
#         self.pam2 = PositionAttentionModule(512)
#         self.pam3 = PositionAttentionModule(1024)
#         self.pam4 = PositionAttentionModule(1024)
#         # self.cam1 = SEModule(channels=1024,reduction=4)
#         # self.cam2 = SEModule(channels=128,reduction=1)
#         # self.sel = SEModule(channels=128,reduction=1)
#
#         self.down1 = nn.Conv2d(1024, 128, kernel_size=1, padding=0, bias=False)
#         self.down2 = nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False)
#         self.down3 = nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)
#         self.down4 = nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=False)
#
#         self.res1 = BasicBlock(128, 128)
#         self.res2 = BasicBlock(256, 256)
#         self.res3 = BasicBlock(512, 512)
#
#         self.fuse = nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=False)
#         self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
#
#         self.fuse3 = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)
#         self.fuse2 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 64, 4, 2, 1)
#         self.finalrelu1 = nn.ReLU()
#         self.finalconv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.finalrelu2 = nn.ReLU()
#         self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1)
#
#         self.seg11 = nn.Conv2d(256, 32, 3, padding=1)
#         self.seg12 = nn.Conv2d(256, 32, 3, padding=1)
#         self.bn11 = nn.BatchNorm2d(128)
#         self.seg21 = nn.Conv2d(512, 32, 3, padding=1)
#         self.seg22 = nn.Conv2d(512, 32, 3, padding=1)
#         self.bn22 = nn.BatchNorm2d(128)
#         self.seg31 = nn.Conv2d(1024, 64, 3, padding=1)
#         self.seg32 = nn.Conv2d(1024, 64, 3, padding=1)
#         self.bn33 = nn.BatchNorm2d(128)
#         self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
#         self.norm1 = nn.BatchNorm2d(256)
#
#         self.deconv2 = nn.Conv2d(256, 128, 3, padding=1)
#         self.norm2 = nn.BatchNorm2d(128)
#
#         self.deconv3 = nn.Conv2d(128, 64, 3, padding=1)
#         self.norm3 = nn.BatchNorm2d(64)
#
#         self.deconv4 = nn.Conv2d(64, 32, 3, padding=1)
#         self.norm4 = nn.BatchNorm2d(32)
#
#         self.deconv5 = nn.Conv2d(32, 32, 3, padding=1)
#         self.norm5 = nn.BatchNorm2d(32)
#
#         self.finalseg = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, input1,input2):
#         # Encoder
#         e11, e12, e13, e14 = self.resnet_features(input1)
#         e21, e22, e23, e24 = self.resnet_features(input2)
#         x_size = input1.size()
#
#         offset1 = self.conv_offset1(torch.cat((e11, e21),dim=1))
#         mask1 = self.conv_mask1(torch.cat((e11, e21),dim=1))
#         # mask1 = torch.sigmoid(mask1)
#         e1 = self.dcn21(torch.cat((e11, e21),dim=1), offset1, mask1)
#
#         # e1 = self.dcn1(torch.cat((e11, e21),dim=1))
#         e1 = self.conv1(e1)
#         # e1 = self.conv1(torch.abs(e11 - e21))  # 128 128 128
#         e1 = self.bn1(e1)
#         e1 = self.rl1(e1)
#
#         offset2 = self.conv_offset2(torch.cat((e12, e22), dim=1))
#         mask2 = self.conv_mask2(torch.cat((e12, e22), dim=1))
#         # mask2 = torch.sigmoid(mask2)
#         e2 = self.dcn22(torch.cat((e12, e22), dim=1),offset2, mask2)
#         e2 = self.conv2(e2)
#         # e2 = self.conv2(torch.abs(e12 - e22))  # 256 64 64
#         e2 = self.bn2(e2)
#         e2 = self.rl2(e2)
#
#         offset3 = self.conv_offset3(torch.cat((e13, e23), dim=1))
#         mask3 = self.conv_mask3(torch.cat((e13, e23), dim=1))
#         # mask3 = torch.sigmoid(mask3)
#         e3 = self.dcn23(torch.cat((e13, e23), dim=1),offset3, mask3)
#         e3 = self.conv3(e3)
#         # e3 = self.conv3(torch.abs(e13 - e23))  # 512 32 32
#         e3 = self.bn3(e3)
#         e3 = self.rl3(e3)
#
#         offset4 = self.conv_offset4(torch.cat((e14, e24), dim=1))
#         mask4 = self.conv_mask4(torch.cat((e14, e24), dim=1))
#         # mask4 = torch.sigmoid(mask4)
#         e4 = self.dcn24(torch.cat((e14, e24), dim=1),offset4, mask4)
#         e4 = self.conv4(e4)
#         # e4 = self.conv4(torch.abs(e14 - e24))  # 1024 16 16
#         e4 = self.bn4(e4)
#         e4 = self.rl4(e4)
#
#         # e4 = self.dblock(e4)
#
#         # d4 = self.decoder4(e4) + e3
#         # d3 = self.decoder3(d4) + e2
#         # d2 = self.decoder2(d3) + e1
#         # d1 = self.decoder1(d2)
#         #
#         # out = self.finaldeconv1(d1)
#         # out = self.finalrelu1(out)
#         # out = self.finalconv2(out)
#         # out = self.finalrelu2(out)
#         # out = self.finalconv3(out)
#
#         d1 = self.deconv1(e4)
#         d1 = self.norm1(d1)
#         d1 = F.interpolate(d1, e3.size()[2:],mode='bilinear')
#
#         d2 = self.deconv2(d1+e3)
#         d2 = self.norm2(d2)
#         d2 = F.interpolate(d2,  e2.size()[2:], mode='bilinear')
#
#         d3 = self.deconv3(d2+e2)
#         d3 = self.norm3(d3)
#         d3 = F.interpolate(d3, e1.size()[2:], mode='bilinear')
#
#         d4 = self.deconv4(d3+e1)
#         d4 = self.norm4(d4)
#         d4 = F.interpolate(d4, x_size[2:], mode='bilinear')
#
#         out = self.finalseg(d4)
#
#         return out

# class BaselineMod(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3, pretrained=False):
#         super(BaselineMod, self).__init__()
#         filters = [64, 128, 256, 512]
#         # filters = [128, 256, 512, 1024]
#         self.resnet_features = resnet34(pretrained=True)
#         # self.dcn1 = DCN(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn2 = DCN(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn3 = DCN(512, 512, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn4 = DCN(1024, 1024, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.resx = BasicBlock(64, 64, stride=1, downsample=None)
#
#         # self.dcn41 = DCN(2048, 2048, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#         # self.dcn42 = DCN(2048, 2048, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
#
#         # self.conv_offset12 = nn.Conv2d(512, 2 * 2 * 3 * 3, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         # self.conv_mask12 = nn.Conv2d(512, 2 * 1 * 3 * 3,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1), bias=True).cuda()
#         # self.conv_offset12.weight.data.zero_()
#         # self.conv_offset12.bias.data.zero_()
#         # self.conv_mask12.weight.data.zero_()
#         # self.conv_mask12.bias.data.zero_()
#         #
#         # self.conv_offset22 = nn.Conv2d(512, 2 * 2 * 3 * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True).cuda()
#         # self.conv_mask22 = nn.Conv2d(512, 2 * 1 * 3 * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=True).cuda()
#         # self.conv_offset22.weight.data.zero_()
#         # self.conv_offset22.bias.data.zero_()
#         # self.conv_mask22.weight.data.zero_()
#         # self.conv_mask22.bias.data.zero_()
#         # self.att4 = BitPosAttMD(512, 8)
#         # self.att3 = BitPosAttMD(256, 4)
#         # self.att2 = BitPosAttMD(128, 2)
#         # self.att1 = BitPosAttMD(64, 1)
#         self.conv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#         # self.c4 = Diff_Block(2048,fusion='all')
#         # self.bn4 = nn.BatchNorm2d(1024)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.rl4 = nn.ReLU()
#
#         self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#         # self.c3 = Diff_Block(1024,fusion='all')
#         # self.bn3 = nn.BatchNorm2d(512)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.rl3 = nn.ReLU()
#         # self.conv_offset1 = nn.Conv2d(128, 36,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         # self.conv_mask1 = nn.Conv2d(128, 18,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=True).cuda()
#         # self.dcn21 = DCNv2(128, 128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         #
#         # self.conv_offset2 = nn.Conv2d(256, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.conv_mask2 = nn.Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.dcn22 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         #
#         # self.conv_offset3 = nn.Conv2d(512, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.conv_mask3 = nn.Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.dcn23 = DCNv2(512, 512, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         #
#         # self.conv_offset4 = nn.Conv2d(1024, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.conv_mask4 = nn.Conv2d(1024, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True).cuda()
#         # self.dcn24 = DCNv2(1024, 1024, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#
#         self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
#         # self.nbk2 = NBlock(256)
#         # self.c2 = Diff_Block(512,fusion='all')
#         # self.bn2 = nn.BatchNorm2d(256)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.rl2 = nn.ReLU()
#         # self.dcn11 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         # self.dcn12 = DCNv2(256, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
#         self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
#         # self.nbk1 = NBlock(128)
#         # self.c1 = Diff_Block(256,fusion='all')
#         # self.bn1 = LayerNorm(128)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.rl1 = nn.ReLU()
#
#         # self.attdiff4 = PosAttDiff(2048)
#         # self.attdiff3 = PosAttDiff(1024)
#         # self.attdiff2 = PosAttDiff(512)
#         # self.attdiff1 = PosAttDiff(256)
#         # self.dblock = Dblock(512)
#         # self.pam1_1 = PositionAttentionModule(64)
#         # self.pam1_2 = PositionAttentionModule(64)
#         # self.pam2 = PositionAttentionModule(512)
#         # self.pam3 = PositionAttentionModule(1024)
#         # self.pam4 = PositionAttentionModule(1024)
#         # self.cam1 = SEModule(channels=1024,reduction=4)
#         # self.cam2 = SEModule(channels=128,reduction=1)
#         # self.sel11 = SEModule(channels=64,reduction=1)
#         # self.sel12 = SEModule(channels=64, reduction=1)
#
#         # self.down1 = nn.Conv2d(1024, 128, kernel_size=1, padding=0, bias=False)
#         # self.down2 = nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False)
#         # self.down3 = nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)
#         # self.down4 = nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=False)
#         #
#         # self.res1 = BasicBlock(128, 128)
#         # self.res2 = BasicBlock(256, 256)
#         # self.res3 = BasicBlock(512, 512)
#         #
#         # self.fuse = nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=False)
#         # self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
#         #
#         # self.fuse3 = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)
#         # self.fuse2 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)
#
#         # self.decoder4 = DecoderBlock(filters[3], filters[2])
#         # self.decoder3 = DecoderBlock(filters[2], filters[1])
#         # self.decoder2 = DecoderBlock(filters[1], filters[0])
#         # self.decoder1 = DecoderBlock(filters[0], filters[0])
#         #
#         # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 64, 4, 2, 1)
#         # self.finalrelu1 = nn.ReLU()
#         # self.finalconv2 = nn.Conv2d(64, 64, 3, padding=1)
#         # self.finalrelu2 = nn.ReLU()
#         # self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1)
#
#         # self.seg11 = nn.Conv2d(256, 32, 3, padding=1)
#         # self.seg12 = nn.Conv2d(256, 32, 3, padding=1)
#         # self.bn11 = nn.BatchNorm2d(128)
#         # self.seg21 = nn.Conv2d(512, 32, 3, padding=1)
#         # self.seg22 = nn.Conv2d(512, 32, 3, padding=1)
#         # self.bn22 = nn.BatchNorm2d(128)
#         # self.seg31 = nn.Conv2d(1024, 64, 3, padding=1)
#         # self.seg32 = nn.Conv2d(1024, 64, 3, padding=1)
#         # self.bn33 = nn.BatchNorm2d(128)
#
#         self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
#         self.norm1 = nn.BatchNorm2d(256)
#
#         self.deconv2 = nn.Conv2d(256, 128, 3, padding=1)
#         self.norm2 = nn.BatchNorm2d(128)
#
#         self.deconv3 = nn.Conv2d(128, 64, 3, padding=1)
#         self.norm3 = nn.BatchNorm2d(64)
#
#         self.deconv4 = nn.Conv2d(64, 32, 3, padding=1)
#         self.norm4 = nn.BatchNorm2d(32)
#
#         # self.deconv5 = nn.Conv2d(32, 32, 3, padding=1)
#         # self.norm5 = nn.BatchNorm2d(32)
#
#         self.finalseg = nn.Conv2d(32, num_classes, 3, padding=1)
#
#         # self.sbn1 = nn.BatchNorm2d(960)
#         self.convblock = nn.Conv2d(960, 64, 3, padding=1)
#         self.sbn2 = nn.BatchNorm2d(64)
#         self.segblock = nn.Conv2d(64, num_classes, 3, padding=1)
#
#     def forward(self, input1,input2):
#         # Encoder
#         e11, e12, e13, e14 = self.resnet_features(input1)
#         e21, e22, e23, e24 = self.resnet_features(input2)
#         x_size = input1.size()
#
#         # offset1 = self.conv_offset1(torch.cat((e11, e21),dim=1))
#         # mask1 = self.conv_mask1(torch.cat((e11, e21),dim=1))
#         # mask1 = torch.sigmoid(mask1)
#         # e1 = self.dcn21(torch.cat((e11, e21),dim=1), offset1, mask1)
#
#         # e1 = self.dcn1(torch.cat((e11, e21),dim=1))
#         # e1 = self.conv1(e1)
#         e1 = self.conv1(torch.abs(e11 - e21))  # 128 128 128
#         e1 = self.bn1(e1)
#         e1 = self.rl1(e1)
#
#         # offset2 = self.conv_offset2(torch.cat((e12, e22), dim=1))
#         # mask2 = self.conv_mask2(torch.cat((e12, e22), dim=1))
#         # mask2 = torch.sigmoid(mask2)
#         # e2 = self.dcn22(torch.cat((e12, e22), dim=1),offset2, mask2)
#         # e2 = self.conv2(e2)
#         e2 = self.conv2(torch.abs(e12 - e22))  # 256 64 64
#         e2 = self.bn2(e2)
#         e2 = self.rl2(e2)
#
#         # offset3 = self.conv_offset3(torch.cat((e13, e23), dim=1))
#         # mask3 = self.conv_mask3(torch.cat((e13, e23), dim=1))
#         # mask3 = torch.sigmoid(mask3)
#         # e3 = self.dcn23(torch.cat((e13, e23), dim=1),offset3, mask3)
#         # e3 = self.conv3(e3)
#         e3 = self.conv3(torch.abs(e13 - e23))  # 512 32 32
#         e3 = self.bn3(e3)
#         e3 = self.rl3(e3)
#
#         # offset4 = self.conv_offset4(torch.cat((e14, e24), dim=1))
#         # mask4 = self.conv_mask4(torch.cat((e14, e24), dim=1))
#         # mask4 = torch.sigmoid(mask4)
#         # e4 = self.dcn24(torch.cat((e14, e24), dim=1),offset4, mask4)
#         # e4 = self.conv4(e4)
#         e4 = self.conv4(torch.abs(e14 - e24))  # 1024 16 16
#         e4 = self.bn4(e4)
#         e4 = self.rl4(e4)
#
#         # e4 = self.dblock(e4)
#
#         se12 = F.interpolate(e12, e11.size()[2:], mode='bilinear',align_corners=False)
#         se13 = F.interpolate(e13, e11.size()[2:], mode='bilinear',align_corners=False)
#         se14 = F.interpolate(e14, e11.size()[2:], mode='bilinear',align_corners=False)
#
#         se22 = F.interpolate(e22, e21.size()[2:], mode='bilinear',align_corners=False)
#         se23 = F.interpolate(e23, e21.size()[2:], mode='bilinear',align_corners=False)
#         se24 = F.interpolate(e24, e21.size()[2:], mode='bilinear',align_corners=False)
#
#         seg1 = self.convblock(torch.cat((e11,se12,se13,se14),dim=1))
#         # seg1 = self.sel11(seg1)
#         seg1 = self.sbn2(seg1)
#         seg1 = F.interpolate(seg1, x_size[2:], mode='bilinear',align_corners=False)
#         seg1 = self.segblock(seg1)
#
#         seg2 = self.convblock(torch.cat((e21, se22, se23, se24), dim=1))
#         # seg2 = self.sel12(seg2)
#         seg2 = self.sbn2(seg2)
#         seg2 = F.interpolate(seg2, x_size[2:], mode='bilinear',align_corners=False)
#         seg2 = self.segblock(seg2)
#
#
#
#         d1 = self.deconv1(e4)
#         d1 = self.norm1(d1)
#         d1 = F.interpolate(d1, e3.size()[2:],mode='bilinear',align_corners=False)
#
#         d2 = self.deconv2(d1+e3)
#         d2 = self.norm2(d2)
#         d2 = F.interpolate(d2,  e2.size()[2:], mode='bilinear',align_corners=False)
#
#         d3 = self.deconv3(d2+e2)
#         d3 = self.norm3(d3)
#         d3 = F.interpolate(d3, e1.size()[2:], mode='bilinear',align_corners=False)
#
#         d4 = self.deconv4(d3+e1)
#         d4 = self.norm4(d4)
#         d4 = F.interpolate(d4, x_size[2:], mode='bilinear',align_corners=False)
#
#         out = self.finalseg(d4)
#
#         return out,seg1,seg2
#         # return out


# class Baseline34(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3, pretrained=False):
#         super(Baseline34, self).__init__()
#         self.resnet_features = resnet34(pretrained=True)
#
#         self.conv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.rl4 = nn.ReLU()
#         # self.lpam4 = Light_PAM(512, 2, 2)
#         #
#         self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.rl3 = nn.ReLU()
#         # self.lpam3 = Light_PAM(256, 4, 4)
#         #
#         self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.rl2 = nn.ReLU()
#         # self.lpam2 = Light_PAM(128, 8, 8)
#         #
#         self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.rl1 = nn.ReLU()
#         # self.lpam1 = Light_PAM(64,16,16)
#
#         # # self.dpam1 = PositionAttentionModule(64)
#         # self.deca1 = ChannelAttentionModule()
#         #
#         # self.dpam2 = PositionAttentionModule(128)
#         # self.deca2 = ChannelAttentionModule()
#         #
#         # self.dpam3 = PositionAttentionModule(256)
#         # self.deca3 = ChannelAttentionModule()
#         #
#         # self.dpam4 = PositionAttentionModule(512)
#         # self.deca4 = ChannelAttentionModule()
#
#         # self.dblock = Dblock(512)
#         #
#         # self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
#         # self.norm1 = nn.BatchNorm2d(256)
#         # self.drl1 = nn.ReLU()
#         # self.deconv2 = nn.Conv2d(256, 128, 3, padding=1)
#         # self.norm2 = nn.BatchNorm2d(128)
#         # self.drl2 = nn.ReLU()
#         # self.deconv3 = nn.Conv2d(128, 64, 3, padding=1)
#         # self.norm3 = nn.BatchNorm2d(64)
#         # self.drl3 = nn.ReLU()
#         # self.deconv4 = nn.Conv2d(64, 32, 3, padding=1)
#         # self.norm4 = nn.BatchNorm2d(32)
#         # self.drl4 = nn.ReLU()
#         # self.finalseg = nn.Conv2d(32, num_classes, 3, padding=1)
#
#         filters = [64, 128, 256, 512]
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], 64)
#         # self.decoder0 = DecoderBlock(32, num_classes)
#         self.finaldeconv1 = nn.Conv2d(64, 32, 3, padding=1)
#         self.finalrelu1 = nn.BatchNorm2d(32)
#         self.finalconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
#         self.finalrelu2 = nn.BatchNorm2d(32)
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#
#     def forward(self, inputs):
#         # Encoder
#         e11, e12, e13, e14 = self.resnet_features(inputs)
#         x_size = inputs.size()
#
#         # e1 = self.deca1(e11)
#         # e1 = self.bn1(e1)
#         # e2 = self.deca2(self.dpam2(e12))
#         # e2 = self.bn2(e2)
#         # e3 = self.deca3(self.dpam3(e13))
#         # e3 = self.bn3(e3)
#         # e4 = self.deca4(self.dpam4(e14))
#         # e4 = self.bn4(e4)
#         e1 = self.conv1(e11)  # 128 128 128
#         e1 = self.bn1(e1)
#         e1 = self.rl1(e1)
#         # e1 = self.lpam1(e1)
#
#         e2 = self.conv2(e12)  # 256 64 64
#         e2 = self.bn2(e2)
#         e2 = self.rl2(e2)
#         # e2 = self.lpam2(e2)
#
#         e3 = self.conv3(e13)  # 512 32 32
#         e3 = self.bn3(e3)
#         e3 = self.rl3(e3)
#         # e3 = self.lpam3(e3)
#
#         e4 = self.conv4(e14)  # 1024 16 16
#         e4 = self.bn4(e4)
#         e4 = self.rl4(e4)
#         # e4 = self.lpam4(e4)
#         # e4 = self.dblock(e4)
#
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         # out = self.decoder0(d1)
#
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return out

class STBCD_Net_SE(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=False):
        super(STBCD_Net_SE, self).__init__()
        self.resnet_features = resnet34(pretrained=True)

        self.conv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        # self.rl4 = nn.ReLU()

        self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        # self.rl3 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        # self.rl2 = nn.ReLU()

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.rl1 = nn.ReLU()

        # self.dblock = Dblock(512)
        # self.fusin = UpFusion()
        self.fusin_mov = UpFusionM()
        self.fusin_new = UpFusionN()
        # self.pam2 = PositionAttentionModule(256)
        # self.pam3 = PositionAttentionModule(512)

        self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        # self.drl1 = nn.ReLU()
        self.deconv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        # self.drl2 = nn.ReLU()
        self.deconv3 = nn.Conv2d(384, 192, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(192)
        # self.drl3 = nn.ReLU()
        self.deconv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        # self.drl4 = nn.ReLU()
        self.finalseg = nn.Conv2d(128, num_classes, 3, padding=1)

        # self.finalnew = nn.Conv2d(32, num_classes, 3, padding=1)
        self.finalmov = nn.Conv2d(128, num_classes, 3, padding=1)

        self.conv_lab11 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        self.bnorm21 = nn.BatchNorm2d(32)
        self.conv_lab21 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.conv_lab1 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.conv_lab2 = nn.Conv2d(32, num_classes, 3, padding=1)


    def forward(self, inputs,labelso):
        # Encoder
        # labels_o{1,3},labels_n{2},labels_m{3},labels{1,2}
        e11, e12, e13, e14 = self.resnet_features(inputs)
        x_size = inputs.size()

        e1 = self.conv1(e11)
        e1 = self.bn1(e1)
        # e1 = self.rl1(e1)
        e2 = self.conv2(e12)
        e2 = self.bn2(e2)
        # e2 = self.rl2(e2)
        e3 = self.conv3(e13)
        e3 = self.bn3(e3)
        # e3 = self.pam2(e3)
        # e3 = self.rl3(e3)
        e4 = self.conv4(e14)
        e4 = self.bn4(e4)
        # e4 = self.pam3(e4)
        # e4 = self.rl4(e4)
        # e4 = self.dblock(e4)
        # fu_new = self.fusin(e1,e2,e3,e4,labelso)
        fu_new = self.fusin_new(e1, e2, e3, e4, labelso)
        fu_mov = self.fusin_mov(e1, e2, e3, e4, labelso)

        d1 = self.deconv1(e4)
        d1 = self.norm1(d1)
        d1 = F.interpolate(d1, e3.size()[2:],mode='bilinear',align_corners=False)

        d2 = self.deconv2(torch.cat([d1,e3],dim=1))
        d2 = self.norm2(d2)
        d2 = F.interpolate(d2,  e2.size()[2:], mode='bilinear',align_corners=False)

        d3 = self.deconv3(torch.cat([d2,e2],dim=1))
        d3 = self.norm3(d3)
        d3 = F.interpolate(d3, e1.size()[2:], mode='bilinear',align_corners=False)

        d4 = self.deconv4(torch.cat([d3, e1],dim=1))
        d4 = self.norm4(d4)
        d4 = F.interpolate(d4, x_size[2:], mode='bilinear',align_corners=False)
        out = self.finalseg(d4)

        # new_out = self.finalnew(d4 *(1-torch.unsqueeze(labelso,dim=1)))
        mov_out = self.finalmov((1 - torch.sigmoid(d4)) * torch.unsqueeze(labelso, dim=1))
        # # mov_out = (1 - torch.sigmoid(out)) * torch.unsqueeze(labelso, dim=1)
        mov_out = self.conv_lab11(fu_mov)
        mov_out = self.bnorm21(mov_out)
        mov_out = self.conv_lab21(mov_out)

        new_out = self.conv_lab1(fu_new)
        new_out = self.bnorm2(new_out)
        new_out = self.conv_lab2(new_out)
        # out_new = self.finalseg2(torch.cat((d4,F.interpolate(fusion, inputs.size()[2:], mode='bilinear',align_corners=False)),dim=1))
        return out, mov_out, new_out


class STBCD_Net_SD(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=False):
        super(STBCD_Net_SD, self).__init__()
        self.resnet_features = resnet34(pretrained=True)

        self.conv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        # self.rl4 = nn.ReLU()

        self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        # self.rl3 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        # self.rl2 = nn.ReLU()

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.rl1 = nn.ReLU()

        # self.dblock = Dblock(512)
        # self.fusin = UpFusion()
        # self.fusin_mov = UpFusionM()
        # self.fusin_new = UpFusionN()
        # self.pam2 = PositionAttentionModule(256)
        # self.pam3 = PositionAttentionModule(512)

        self.deconv1 = nn.Conv2d(512,256, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        # self.drl1 = nn.ReLU()
        self.deconv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        # self.drl2 = nn.ReLU()
        self.deconv3 = nn.Conv2d(384, 192, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(192)
        # self.drl3 = nn.ReLU()
        self.deconv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        # self.drl4 = nn.ReLU()
        self.finalseg = nn.Conv2d(128, num_classes, 3, padding=1)

        self.finalnew = nn.Conv2d(128, num_classes, 3, padding=1)
        self.finalmov = nn.Conv2d(128, num_classes, 3, padding=1)

        # self.conv_lab11 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        # self.bnorm21 = nn.BatchNorm2d(32)
        # self.conv_lab21 = nn.Conv2d(32, num_classes, 3, padding=1)
        #
        # self.conv_lab1 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        # self.bnorm2 = nn.BatchNorm2d(32)
        # self.conv_lab2 = nn.Conv2d(32, num_classes, 3, padding=1)


    def forward(self, inputs,labelso):
        # Encoder
        # labels_o{1,3},labels_n{2},labels_m{3},labels{1,2}
        e11, e12, e13, e14 = self.resnet_features(inputs)
        x_size = inputs.size()

        e1 = self.conv1(e11)
        e1 = self.bn1(e1)
        # e1 = self.rl1(e1)
        e2 = self.conv2(e12)
        e2 = self.bn2(e2)
        # e2 = self.rl2(e2)
        e3 = self.conv3(e13)
        e3 = self.bn3(e3)
        # e3 = self.pam2(e3)
        # e3 = self.rl3(e3)
        e4 = self.conv4(e14)
        e4 = self.bn4(e4)
        # e4 = self.pam3(e4)
        # e4 = self.rl4(e4)
        # e4 = self.dblock(e4)
        # fu_new = self.fusin(e1,e2,e3,e4,labelso)
        # fu_new = self.fusin_new(e1, e2, e3, e4, labelso)
        # fu_mov = self.fusin_mov(e1, e2, e3, e4, labelso)

        d1 = self.deconv1(e4)
        d1 = self.norm1(d1)
        d1 = F.interpolate(d1, e3.size()[2:],mode='bilinear',align_corners=False)

        d2 = self.deconv2(torch.cat([d1,e3],dim=1))
        d2 = self.norm2(d2)
        d2 = F.interpolate(d2,  e2.size()[2:], mode='bilinear',align_corners=False)

        d3 = self.deconv3(torch.cat([d2,e2],dim=1))
        d3 = self.norm3(d3)
        d3 = F.interpolate(d3, e1.size()[2:], mode='bilinear',align_corners=False)

        d4 = self.deconv4(torch.cat([d3, e1],dim=1))
        d4 = self.norm4(d4)
        d4 = F.interpolate(d4, x_size[2:], mode='bilinear',align_corners=False)
        out = self.finalseg(d4)


        mov_out = self.finalmov((1 - torch.sigmoid(d4)) * torch.unsqueeze(labelso, dim=1))
        # mov_out = 1 - self.finalmov(d4 * torch.unsqueeze(labelso, dim=1))
        new_out = self.finalnew(d4 * torch.unsqueeze(1-labelso, dim=1))

        return out, mov_out, new_out