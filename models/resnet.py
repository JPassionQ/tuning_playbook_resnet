import torch
import torch.nn as nn

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")

# BasicBlock for ResNet-18/34
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, activation='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True,track_running_stats=True)
        self.activation = activation
        self.act1 = get_activation(activation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True,track_running_stats=True)
        self.act2 = get_activation(activation)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act2(out)
        return out

# Bottleneck for ResNet-50/101/152
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None, activation='relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = activation
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        self.act3 = get_activation(activation)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act3(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dropout_p=0.5, activation='relu'):
        """
        block: BasicBlock 或 Bottleneck，表示残差块的类型
        layers: 整数列表，每个 ResNet 层中包含的残差块数量，例如 [2, 2, 2, 2] 表示 ResNet-18 的配置
        dropout_p: Dropout概率，默认0.5
        activation: 激活函数类型，支持 'relu', 'leaky_relu', 'gelu', 'tanh', 'silu'
        """
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.activation = activation
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = get_activation(activation)

        self.layer1 = self._make_layer(block, 64, layers[0], activation=activation)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation=activation)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, activation='relu'):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, activation=activation))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, activation=activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def resnet18(num_classes=1000, dropout_p=0.5, activation='relu'):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dropout_p=dropout_p, activation=activation)

def resnet34(num_classes=1000, dropout_p=0.5, activation='relu'):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, dropout_p=dropout_p, activation=activation)

def resnet50(num_classes=1000, dropout_p=0.5, activation='relu'):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, dropout_p=dropout_p, activation=activation)

def resnet101(num_classes=1000, dropout_p=0.5, activation='relu'):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, dropout_p=dropout_p, activation=activation)

def resnet152(num_classes=1000, dropout_p=0.5, activation='relu'):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, dropout_p=dropout_p, activation=activation)
