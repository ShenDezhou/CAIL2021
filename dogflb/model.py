from jittor.models import Resnet50, AlexNet
import jittor.nn as nn 

class Net(nn.Module):
    def __init__(self, num_classes):
        self.base_net = Resnet50(pretrained=True, num_classes=num_classes)
        # self.fc = nn.Linear(1000, num_classes)
    
    def execute(self, x):
        x = self.base_net(x)
        # x = self.fc(x)
        return x


class NetA(nn.Module):
    def __init__(self, num_classes):
        self.base_net = AlexNet()
        self.fc = nn.Linear(1000, num_classes)

    def execute(self, x):
        x = self.base_net(x)
        x = self.fc(x)
        return x