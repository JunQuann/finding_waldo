from skimage.feature import hog
from helper import convert
import numpy as np
import cv2


class FeatureGenerator:
    def __init__(self, color_model, hog_params):
        self.color_model = color_model
        self.hog_params = hog_params
        self.hogA, self.hogB, self.hogC = None, None, None
        self.hogA_img, self.hogB_img, self.hogC_img = None, None, None
        self.img = None
        self.RGB_img = None

    def get_hog(self, channel):
        features = hog(channel, **self.hog_params)
        return features

    def generate_features(self, img):
        self.img = convert(img, dest_model=self.color_model)
        self.RGB_img = convert(img, dest_model="rgb")

        self.hogA, self.hogA_img = self.get_hog(self.img[:, :, 0])
        self.hogB, self.hogB_img = self.get_hog(self.img[:, :, 1])
        self.hogC, self.hogC_img = self.get_hog(self.img[:, :, 2])
        hog = np.hstack((self.hogA, self.hogB, self.hogC))

        return hog

    def visualize(self):
        return self.RGB_img, self.hogA_img, self.hogB_img, self.hogC_img
