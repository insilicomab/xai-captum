import torch
import timm


# class to load model weights in pth format
class LoadPthModel():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_pretrained(self, timm_name, model_dir):
        model = timm.create_model(timm_name, pretrained=False, num_classes=self.num_classes)
        model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

        return model


# function to load model weights in pth format
def load_model_pth(timm_name: str, num_classes: int, model_dir: str):
    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    return model