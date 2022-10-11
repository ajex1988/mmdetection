from .mobilenet_v2 import MobileNetV2

from ..builder import BACKBONES

@BACKBONES.register_module()
class MobileNetV2R(MobileNetV2):
    def __init__(self, arch_settings):
        MobileNetV2.arch_settings = arch_settings
        super(MobileNetV2R, self).__init__()