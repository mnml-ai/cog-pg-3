#@title utils
import torch
from typing import List
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler
)
from controlnet_aux import (
    HEDdetector,
    LineartDetector,
    CannyDetector,
    MLSDdetector,
)
from transformers import pipeline

def resize_image(image, max_width, max_height):
    """
    Resize an image to a specific height while maintaining the aspect ratio and ensuring
    that neither width nor height exceed the specified maximum values.

    Args:
        image (PIL.Image.Image): The input image.
        max_width (int): The maximum allowable width for the resized image.
        max_height (int): The maximum allowable height for the resized image.

    Returns:
        PIL.Image.Image: The resized image.
    """
    # Get the original image dimensions
    original_width, original_height = image.size

    # Calculate the new dimensions to maintain the aspect ratio and not exceed the maximum values
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height

    # Choose the smallest ratio to ensure that neither width nor height exceeds the maximum
    resize_ratio = min(width_ratio, height_ratio)

    # Calculate the new width and height
    new_width = int(original_width * resize_ratio)
    new_height = int(original_height * resize_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

def sort_dict_by_string(input_string, your_dict):
    if not input_string or not isinstance(input_string, str):
        # Return the original dictionary if the string is empty or not a string
        return your_dict

    order_list = [item.strip() for item in input_string.split(',')]

    # Include keys from the input string that are present in the dictionary
    valid_keys = [key for key in order_list if key in your_dict]

    # Include keys from the dictionary that are not in the input string
    remaining_keys = [key for key in your_dict if key not in valid_keys]

    sorted_dict = {key: your_dict[key] for key in valid_keys}
    sorted_dict.update({key: your_dict[key] for key in remaining_keys})

    return sorted_dict

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "KLMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
    "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
     "DPM++ 3M SDE Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
}


def depth_preprocess(self, img):
        image = self.detectors["depth"](img)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

def scribble_preprocess(self, img):
    return self.detectors["scribble"](img, scribble=True)

def mlsd_preprocess(self, img):
    return self.detectors["mlsd"](img)

def canny_preprocess(self, img):
    return self.detectors["canny"](img)

def lineart_preprocess(self, img):
    return self.detectors["lineart"](img, coarse= False)

def tile_preprocess(self, img):
    return img

def brightness_preprocess(self, img):
    return img

def inpaint_preprocess(self, image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    # Convert the torch tensor back to a Pillow image
    # image_pil = Image.fromarray((image.squeeze().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

    return image

PROCESSORS_CACHE = "processors-cache"

AUX_IDS = {
    "depth": {
        "path": "fusing/stable-diffusion-v1-5-controlnet-depth",
        "xl_path": "diffusers/controlnet-depth-sdxl-1.0-small",
        "t2i_adapter_xl_path": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
        "detector": lambda: pipeline('depth-estimation'),
        "preprocessor": depth_preprocess
    },
    'lineart': {
        "path": "ControlNet-1-1-preview/control_v11p_sd15_lineart",
        "xl_path": "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
        "t2i_adapter_xl_path": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
        "detector": lambda: LineartDetector.from_pretrained("lllyasviel/Annotators"),
        "preprocessor": lineart_preprocess
    },
    'mlsd': {
        "path": "lllyasviel/control_v11p_sd15_mlsd",
        "detector": lambda: MLSDdetector.from_pretrained("lllyasviel/Annotators"),
        "preprocessor": mlsd_preprocess
    },
    'canny': {
        "path": "lllyasviel/control_v11p_sd15_canny",
        "detector": lambda: CannyDetector(),
        "preprocessor": canny_preprocess
    },
    "inpainting": {
        "path": "lllyasviel/control_v11p_sd15_inpaint",
        "detector": lambda: None,
        "preprocessor": inpaint_preprocess
    },
}