from jittor.models import Resnet50, resnet152,resnet34, alexnet,googlenet,densenet121,densenet169, densenet201, inception_v3,mnasnet0_5,mobilenet_v2, shufflenet_v2_x0_5, squeezenet1_0, vgg11
from seresnet import resnet50 as seresnet50
from seresnet import resnet152 as seresnet152
from seresnet import resnet200 as seresnet200
from cbamresnet import resnet152 as caresnet152
from searesnet import resnet152 as searesnet152
from aresnet import resnet152 as aresnet152
from gresnet import resnet152 as gresnet152
import jittor.nn as nn
import jittor as jt

class Net1(nn.Module):
    def __init__(self, num_classes):
        self.base_net = Resnet50(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x


class Net1_z(nn.Module):
    def __init__(self, num_classes):
        self.base_net = resnet152(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net1_d(nn.Module):
    def __init__(self, num_classes):
        self.base_net = resnet152(pretrained=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.drop(x)
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

class Net4_y(nn.Module):
    def __init__(self, num_classes):
        self.base_net = densenet169(pretrained=True)
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

class Net10_z(nn.Module):
    def __init__(self, num_classes):
        self.base_net = seresnet152(pretrained=True)
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


class Net12(nn.Module):
    def __init__(self, num_classes):
        self.base_net = caresnet152(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net13(nn.Module):
    def __init__(self, num_classes):
        self.base_net = searesnet152(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net14(nn.Module):
    def __init__(self, num_classes):
        self.base_net = aresnet152(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x


class Net15(nn.Module):
    def __init__(self, num_classes):
        self.base_net = gresnet152(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x

class Net16(nn.Module):
    def __init__(self, num_classes):
        self.base_net = seresnet152(pretrained=True)
        self.comp_net = resnet152(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x1 = self.base_net(x)
        x2 = self.comp_net(x)
        x = x1 * x2
        x = self.fc(x)
        return x

class Net16_z(nn.Module):
    def __init__(self, num_classes):
        self.base_net = seresnet152(pretrained=True)
        self.comp_net = resnet152(pretrained=True)
        self.fc = nn.Linear(3*1000, num_classes)

    def rep_aug(self, x1, x2):
        a = x1 * x2
        b = x1 + x2
        d = x1 - x2
        x = jt.contrib.concat([a,b,d], dim=1)
        return x


    def execute(self, x):
        x1 = self.base_net(x)
        x2 = self.comp_net(x)
        x = self.rep_aug(x1, x2)
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self, num_classes):
        self.base_net = seresnet200(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x