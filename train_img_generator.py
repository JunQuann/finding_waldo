from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2


class TrainImageGenerator:
    """
    target_dict = {
        waldo: {
            index: [bbox...]
        }
    }
    """

    IMG_PATH = "datasets/JPEGImages/{}.jpg"
    TRAIN_IMG_DIR = "datasets/{}"
    POSITIVE_TRAIN_IMG_DIR = "datasets/{}/positive"
    NEGATIVE_TRAIN_IMG_DIR = "datasets/{}/negative"
    IMG_DIM = (32, 64)

    def __init__(self, target, target_dict):
        self.target = target
        self.target_dict = target_dict
        self.positive_path, self.negative_path = self.create_dir()

    def generate_training_images(self):
        for training_img_index, bboxes in self.target_dict.items():
            img_path = TrainImageGenerator.IMG_PATH.format(training_img_index)
            img = cv2.imread(img_path)
            self.generate_positive_images(img, bboxes, training_img_index)
            self.generate_negative_images(img, bboxes, training_img_index)

    def generate_positive_images(self, img, bboxes, training_img_index):
        for index, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            labeled_img = img[y_min:y_max, x_min:x_max]
            rescaled_img = cv2.resize(
                labeled_img, TrainImageGenerator.IMG_DIM, interpolation=cv2.INTER_CUBIC
            )
            cv2.imwrite(
                os.path.join(
                    self.positive_path, "{}_{}.jpg".format(training_img_index, index)
                ),
                rescaled_img,
            )

    def generate_negative_images(self, img, bboxes, training_img_index):
        img_height, img_width, img_channel = img.shape

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            bbox_height = y_max - y_min
            bbox_width = x_max - x_min
            img[y_min:y_max, x_min:x_max] = np.zeros(
                [bbox_height, bbox_width, img_channel]
            )

        negative_img_count = 0
        x_split_imgs = np.array_split(img, 10, axis=1)
        for img in x_split_imgs:
            negative_images = np.array_split(img, 10, axis=0)
            for negative_img in negative_images:
                rescaled_img = cv2.resize(
                    negative_img,
                    TrainImageGenerator.IMG_DIM,
                    interpolation=cv2.INTER_CUBIC,
                )
                cv2.imwrite(
                    os.path.join(
                        self.negative_path,
                        "{}_{}.jpg".format(training_img_index, negative_img_count),
                    ),
                    rescaled_img,
                )
                negative_img_count += 1

    def create_dir(self):
        if not os.path.isdir(TrainImageGenerator.TRAIN_IMG_DIR.format(self.target)):
            os.mkdir(TrainImageGenerator.TRAIN_IMG_DIR.format(self.target))
            os.mkdir(TrainImageGenerator.POSITIVE_TRAIN_IMG_DIR.format(self.target))
            os.mkdir(TrainImageGenerator.NEGATIVE_TRAIN_IMG_DIR.format(self.target))
        return (
            TrainImageGenerator.POSITIVE_TRAIN_IMG_DIR.format(self.target),
            TrainImageGenerator.NEGATIVE_TRAIN_IMG_DIR.format(self.target),
        )
