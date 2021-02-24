from jittor.models import Resnet50, alexnet,googlenet,densenet121,inception_v3,mnasnet0_5,mobilenet_v2, shufflenet_v2_x0_5, squeezenet1_0, vgg11
from seresnet import resnet50 as seresnet50
import jittor.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes):
        self.base_net = Resnet50(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x


class Net2(nn.Module):
    def __init__(self, num_classes):
        self.base_net = alexnet(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net3(nn.Module):
    def __init__(self, num_classes):
        self.base_net = googlenet(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x


class Net4(nn.Module):
    def __init__(self, num_classes):
        self.base_net = densenet121(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net5(nn.Module):
    def __init__(self, num_classes):
        self.base_net = inception_v3(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net6(nn.Module):
    def __init__(self, num_classes):
        self.base_net = mnasnet0_5(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net7(nn.Module):
    def __init__(self, num_classes):
        self.base_net = mobilenet_v2(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x


class Net8(nn.Module):
    def __init__(self, num_classes):
        self.base_net = shufflenet_v2_x0_5(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net9(nn.Module):
    def __init__(self, num_classes):
        self.base_net = squeezenet1_0(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net10(nn.Module):
    def __init__(self, num_classes):
        self.base_net = seresnet50(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net11(nn.Module):
    def __init__(self, num_classes):
        self.base_net = vgg11(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x