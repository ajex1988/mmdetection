import torch.nn as nn
import torch.nn.functional as F
from ..builder import BACKBONES


@BACKBONES.register_module()
class FCRNBNRP(nn.Module):
    """
    Parameterized Reduced version
    In each layer the number of filters reduced to 1/n where n is the parameter
    """
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.C1 = nn.Conv2d(3, int(64/self.n), 5, padding=2)
        self.C2 = nn.Conv2d(int(64/self.n), int(128/self.n), 5, padding=2)
        self.C3 = nn.Conv2d(int(128/self.n), int(128/self.n), 3, padding=1)
        self.C4 = nn.Conv2d(int(128/self.n), int(128/self.n), 3, padding=1)
        self.C5 = nn.Conv2d(int(128/self.n), int(256/self.n), 3, padding=1)
        self.C6 = nn.Conv2d(int(256/self.n), int(256/self.n), 3, padding=1)
        self.C7 = nn.Conv2d(int(256/self.n), int(512/self.n), 3, padding=1)
        self.C8 = nn.Conv2d(int(512/self.n), int(512/self.n), 3, padding=1)
        self.C9 = nn.Conv2d(int(512/self.n), int(512/self.n), 3, padding=1)

        self.MP = nn.MaxPool2d(2, 2)


    def forward(self, x):
        out1 = F.relu(self.C1(x))
        out1 = self.MP(out1)

        out2 = F.relu(self.C2(out1))
        out2 = self.MP(out2)

        out3 = F.relu(self.C3(out2))
        out3 = F.relu(self.C4(out3))
        out3 = self.MP(out3)

        out4 = F.relu(self.C5(out3))
        out4 = F.relu(self.C6(out4))
        out4 = self.MP(out4)

        out5 = F.relu(self.C7(out4))
        out5 = F.relu(self.C8(out5))
        out5 = F.relu(self.C9(out5))

        return (out1, out3, out4, out5)
