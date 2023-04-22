import imgaug
import numpy as np

from concern.config import State
from .data_process import DataProcess
from data.augmenter import AugmenterBuilder
import cv2
import math

import random

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class AugmentData(DataProcess):
    augmenter_args = State(autoload=False)
    using_resize_v2 = State(autoload=False)

    def __init__(self, **kwargs):
        self.augmenter_args = kwargs.get('augmenter_args')
        self.keep_ratio = kwargs.get('keep_ratio')
        self.only_resize = kwargs.get('only_resize')
        self.using_resize_v2 = kwargs.get('using_resize_v2', False)
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    def resize_image_v2(self, im): 
        ''' 
        resize image to a size multiple of 32 which is required by the network 
        :param im: the resized image 
        :param max_side_len: limit of max image size to avoid out of memory in gpu 
        :return: the resized image and the resize ratio 
        ''' 
        min_side_len= 1024 # 2048 # 1600 # 
        max_side_len= 1600 # 3096 # 2048 # 
        h, w, _ = im.shape 
        resize_w = w 
        resize_h = h 
        # limit the max side 
        if min(resize_h, resize_w) < min_side_len: 
            min_ratio = float(min_side_len) / resize_h if resize_h < resize_w else float(min_side_len) / resize_w 
        else: 
            min_ratio = 1. 
        resize_h = int(resize_h * min_ratio) 
        resize_w = int(resize_w * min_ratio) 
        if max(resize_h, resize_w) > max_side_len: 
            max_ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w 
        else: 
            max_ratio = 1. 
        resize_h = int(resize_h * max_ratio) 
        resize_w = int(resize_w * max_ratio) 
        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32 
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32 
        # print 'h:%d,w:%d' % (resize_h, resize_w) 
        im = cv2.resize(im, (int(resize_w), int(resize_h))) 
        return im 

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def process(self, data):
        image = data['image']
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                if self.using_resize_v2:
                    data['image'] = self.resize_image_v2(image)
                else:
                    data['image'] = self.resize_image(image)
            else:
                data['image'] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        if not self.only_resize:
            data['is_training'] = True 
        else:
            data['is_training'] = False 
        return data


class AugmentDetectionData(AugmentData):
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',
                'text': line['text'],
            })
            # for oriented text eval in tt
            if False:
                if len(new_poly) <= 4:
                    line_polys[-1]['ignore'] = True
        if 'char_lines' in data:
            char_line_box = []
            for char_line in data['char_lines']:
                if self.only_resize:
                    new_char_box = [(p[0], p[1]) for p in char_line]
                else:
                    new_char_box = self.may_augment_poly(aug, shape, char_line)
                char_line_box.append(new_char_box)
            data['char_lines'] = char_line_box
        if 'char_label_additional_mask_boxes' in data:
            char_line_box = []
            for char_line in data['char_label_additional_mask_boxes']:
                if self.only_resize:
                    new_char_box = [(p[0], p[1]) for p in char_line]
                else:
                    new_char_box = self.may_augment_poly(aug, shape, char_line)
                char_line_box.append(new_char_box)
            data['char_label_additional_mask_boxes'] = char_line_box
        data['polys'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly


class RandomBrightness(object):
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            brightness_factor = random.uniform(0.5, 2)
            image = F.adjust_brightness(image, brightness_factor)
        return image


class RandomContrast(object):
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            contrast_factor = random.uniform(0.5, 2)
            image = F.adjust_contrast(image, contrast_factor)
        return image

class RandomHue(object):
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            hue_factor = random.uniform(-0.25, 0.25)
            image = F.adjust_hue(image, hue_factor)
        return image


class RandomSaturation(object):
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            saturation_factor = random.uniform(0.5, 2)
            image = F.adjust_saturation(image, saturation_factor)
        return image


class RandomGamma(object):
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            gamma_factor = random.uniform(0.5, 2)
            image = F.adjust_gamma(image, gamma_factor)
        return image


class AugmentImageData(DataProcess):
    use_color = State(default=True)
    use_resize = State(default=False)

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        self.use_color = bool(self.use_color)
        self.use_resize = bool(self.use_resize)
        self.brightness = RandomBrightness()
        self.contrast = RandomContrast()
        self.gamma = RandomGamma()
        self.saturation = RandomSaturation()
        self.hue = RandomHue()

    def process(self, data):
        image = data['image']
        pil_image = Image.fromarray(image.astype('uint8'))
        if self.use_color:
            pil_image = self.brightness(pil_image)
            pil_image = self.contrast(pil_image)
            pil_image = self.gamma(pil_image)
            pil_image = self.saturation(pil_image)
            pil_image = self.hue(pil_image)
            image = np.array(pil_image).astype('float32')

        data['image'] = image
        return data
