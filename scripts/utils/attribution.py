import captum
from abc import ABC, abstractmethod


# Abstract Base Class
class CaptumAlgorithm(ABC):
    @abstractmethod
    def attribution(self, input_img, target):
        pass


# Guided GradCAM
class GuidedGradCamAlgorithm(CaptumAlgorithm):
    def __init__(self, model, layer, device_ids=None):
        self._guided_gc = captum.attr.GuidedGradCam(model, layer, device_ids)
    
    def attribution(self, input_img, target, additional_forward_args=None, interpolate_mode='nearest', attribute_to_layer_input=False):
        return self._guided_gc.attribute(input_img, target, additional_forward_args, interpolate_mode, attribute_to_layer_input)







