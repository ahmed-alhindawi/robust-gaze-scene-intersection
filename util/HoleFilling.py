#!/usr/bin/env python

from enum import Enum
from functools import partial

import cv2
import numpy as np


class HoleFillingFilter(object):
    class FillMode(Enum):
        fast_matching = 5,
        naiver_stokes = 6

    """
    Original Kinect depth image has many invalid pixels (black hole).
    This function helps you to fill the invalid pixel with the proper value.
    Adapted from https://github.com/intelligent-control-lab/Kinect_Smoothing
    """

    def __init__(self, fill_mode: FillMode = FillMode.naiver_stokes, radius: float = 5.0, min_valid_depth: float = 0.1, max_valid_depth: float = 20):
        """
        :param fill_mode: enum, specific methods for hole filling.
        :param radius: float, radius of the neighboring area used for fill in a hole
        :param min_valid_depth: float,  a depth pixel is considered as invalid value, when depth < min_valid_depth
        :param max_valid_depth: float,  a depth pixel is considered as invalid value, when depth > max_valid_depth
        """
        self.valid_depth_min = min_valid_depth
        self.valid_depth_max = max_valid_depth
        if fill_mode == HoleFillingFilter.FillMode.fast_matching:
            inpaint_fn = partial(cv2.inpaint, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        elif fill_mode == HoleFillingFilter.FillMode.naiver_stokes:
            inpaint_fn = partial(cv2.inpaint, inpaintRadius=radius, flags=cv2.INPAINT_NS)
        else:
            raise NotImplementedError("No other inpainting hole filling method is implemented")

        self._inpaint_fn = inpaint_fn

    def _inpainting_smoothing(self, image: np.ndarray) -> np.ndarray:
        """
        smoothing image with inpainting method, such as FMI, NS
        :param image: numpy-array,
        :return: smoothed: numpy-array, smoothed image
        """
        image[image <= self.valid_depth_min] = 0
        image[image >= self.valid_depth_max] = 0
        image[np.isnan(image)] = 0
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[image == 0] = 1
        smoothed = self._inpaint_fn(image, mask[:, :, np.newaxis])
        return smoothed

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        smooth the image using specific method
        :param image: numpy-array,
        :return: smoothed_image: numpy-array,
        """
        image = image.copy()
        smoothed_image = self._inpainting_smoothing(image)
        return smoothed_image
