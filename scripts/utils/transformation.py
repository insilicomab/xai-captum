import torchvision
from torchvision import transforms


class DataTransform():

    def __init__(self, photo_size):
        
        self.transform = {
            'input': transforms.Compose([
                transforms.Resize([photo_size, photo_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
            'original': transforms.Compose([
                transforms.Resize([photo_size, photo_size]),
                ]),
        }
    
    def __call__(self, phase, img):
        """
        Parameters
        ----------
        phase : 'input' or 'original'
        """
        return self.transform[phase](img)