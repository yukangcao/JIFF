# code borrowed from https://github.com/jiuxianghedonglu/AnimeHeadDetection/blob/master/detect_image.py
import random

import torch
from PIL import Image
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class RandomBrightness(object):
    def __init__(self, brightness=0.15):
        self.brightness = [max(0, 1-brightness), 1+brightness]

    def __call__(self, image, target):
        factor = random.uniform(self.brightness[0], self.brightness[1])
        img = F.adjust_brightness(image, factor)
        return img, target


class RandomContrast(object):
    def __init__(self, contrast=0.15):
        self.contrast = [max(0, 1-contrast), 1+contrast]

    def __call__(self, image, target):
        factor = random.uniform(self.contrast[0], self.contrast[1])
        img = F.adjust_contrast(image, factor)
        return img, target


class RandomSaturation(object):
    def __init__(self, saturation=0.15):
        self.saturation = [max(0, 1-saturation), 1+saturation]

    def __call__(self, image, target):
        factor = random.uniform(self.saturation[0], self.saturation[1])
        img = F.adjust_saturation(image, factor)
        return img, target


class RandomHue(object):
    def __init__(self, hue=0.075):
        self.hue = [-hue, hue]

    def __call__(self, image, target):
        factor = random.uniform(self.hue[0], self.hue[1])
        img = F.adjust_hue(image, factor)
        return img, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transforms(train_flag=True):
    transforms = []
    if train_flag:
        transforms += [
            RandomBrightness(),
            RandomContrast(),
            RandomSaturation(),
            RandomHue()
        ]
    transforms.append(ToTensor())
    if train_flag:
        transforms.append(RandomHorizontalFlip(prob=0.5))
    return Compose(transforms)


if __name__ == '__main__':
    pass
