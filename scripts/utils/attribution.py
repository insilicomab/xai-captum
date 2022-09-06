import captum


class GeneralAttribution():

    def __init__(self, input_img, original_img):
        self.input_img = input_img
        self.original_img = original_img