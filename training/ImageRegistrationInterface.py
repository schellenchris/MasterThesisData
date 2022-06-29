from abc import ABC, abstractmethod

import SimpleITK as sitk
import numpy as np

class ImageRegistrationInterface(ABC):
    @abstractmethod
    def register_images(moving_image: sitk.Image, fixed_image: sitk.Image):
        pass