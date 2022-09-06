import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms


# class to load input images and processed original images
class ImageLoader():
    
    def __init__(self, img_dir: str, photo_size: int):
        self.img_dir = img_dir
        self.photo_size = photo_size
        self.transform = {
            'input': transforms.Compose([
                transforms.Resize([self.photo_size, self.photo_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
            'original': transforms.Compose([
                transforms.Resize([self.photo_size, self.photo_size]),
                ]),
        }

    
    def process_images(self):

        # read an image file
        img = Image.open(self.img_dir).convert('RGB')

        # process input image
        input_img = self.transform['input'](img).unsqueeze(0)

        # process original image
        original_img = np.array(self.transform['original'](img))

        return input_img, original_img