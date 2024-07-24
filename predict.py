import os
import re
import time
from typing import List, Dict
import requests
from urllib.parse import urlparse

import torch
import numpy as np
from PIL import Image
import cv2

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
)
from transformers import CLIPVisionModelWithProjection
from compel import Compel
from cog import BasePredictor, Input, Path

from generator import Generator
from utils import SCHEDULERS, resize_image

class Predictor(BasePredictor):
    def setup(self):
        """Initialize the predictor."""
        self.loaded_models = {}
        os.makedirs("model_cache", exist_ok=True)

    def _download_model(self, url: str) -> str:
        """Download a model file from a URL."""
        local_filename = os.path.join("model_cache", os.path.basename(urlparse(url).path))
        
        if os.path.exists(local_filename):
            print(f"Model file already exists: {local_filename}")
            return local_filename

        print(f"Downloading model from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return local_filename

    def _load_custom_model(self, model_url: str):
        """Load a custom model from a URL."""
        if model_url in self.loaded_models:
            return self.loaded_models[model_url]

        model_path = self._download_model(model_url)
        print(f"Loading custom model from {model_path}")
        try:
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to("cuda")
            self.loaded_models[model_url] = pipe
            return pipe
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            raise ValueError(f"Failed to load custom model from {model_url}")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt - using compel, use +++ to increase words weight:: doc: https://github.com/damian0815/compel/tree/main/doc || https://invoke-ai.github.io/InvokeAI/features/PROMPTS/#attention-weighting",),
        negative_prompt: str = Input(
            description="Negative prompt - using compel, use +++ to increase words weight//// negative-embeddings available ///// FastNegativeV2 , boring_e621_v4 , verybadimagenegative_v1 || to use them, write their keyword in negative prompt",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        num_inference_steps: int = Input(description="Steps to run denoising", default=20),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.0,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=None),
        eta: float = Input(
            description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise",
            default=0.0,
        ),
        guess_mode: bool = Input(
            description="In this mode, the ControlNet encoder will try best to recognize the content of the input image even if you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.",
            default=False,
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        num_outputs: int = Input(
            description="Number of images to generate",
            ge=1,
            le=10,
            default=1,
        ),
        max_width: int = Input(
            description="Max width/Resolution of image",
            default=512,
        ),
        max_height: int = Input(
            description="Max height/Resolution of image",
            default=512,
        ),
        scheduler: str = Input(
            default="DDIM",
            choices=SCHEDULERS.keys(),
            description="Choose a scheduler.",
        ),
        lineart_image: Path = Input(
            description="Control image for lineart controlnet", default=None
        ),
        lineart_conditioning_scale: float = Input(
            description="Conditioning scale for lineart controlnet",
            default=1,
        ),
        depth_image: Path = Input(
            description="Control image for depth controlnet", default=None
        ),
        depth_conditioning_scale: float = Input(
            description="Conditioning scale for depth controlnet",
            default=1,
        ),
        canny_image: Path = Input(
            description="Control image for canny controlnet", default=None
        ),
        canny_conditioning_scale: float = Input(
            description="Conditioning scale for canny controlnet",
            default=1,
        ),
        mlsd_image: Path = Input(
            description="Control image for mlsd controlnet", default=None
        ),
        mlsd_conditioning_scale: float = Input(
            description="Conditioning scale for mlsd controlnet",
            default=1,
        ),
        inpainting_image: Path = Input(
            description="Control image for inpainting controlnet", default=None
        ),
        mask_image: Path = Input(
            description="mask image for inpainting controlnet", default=None
        ),
        positive_auto_mask_text: str = Input(
            description="comma seperated list of objects for mask, AI will auto create mask of these objects, if mask text is given, mask image will not work", default=None
        ),
        negative_auto_mask_text: str = Input(
            description="comma seperated list of objects you dont want to mask, AI will auto delete these objects from mask, only works if positive_auto_mask_text is given", default=None
        ),
        inpainting_conditioning_scale: float = Input(
            description="Conditioning scale for inpainting controlnet",
            default=1,
        ),
        sorted_controlnets: str = Input(
            description="Comma seperated string of controlnet names, list of names: tile, inpainting, lineart,depth ,scribble , brightness /// example value: tile, inpainting, lineart ", default="tile, inpainting, lineart"
        ),
        ip_adapter_ckpt: str = Input(
            description="IP Adapter checkpoint", default="ip-adapter_sd15.bin", choices=["ip-adapter_sd15.bin", "ip-adapter-plus_sd15.bin", "ip-adapter-plus-face_sd15.bin"]
        ),
        ip_adapter_image: Path = Input(
            description="IP Adapter image", default=None
        ),
        ip_adapter_weight: float = Input(
            description="IP Adapter weight", default=1.0, 
        ),
        img2img_image: Path = Input(
            description="Image2image image", default=None
        ),
        img2img_strength: float = Input(
            description="img2img strength, does not work when inpainting image is given, 0.1-same image, 0.99-complete destruction of image", default=0.5,
        ),
        add_more_detail_lora_scale: float = Input(
            description="Scale/ weight of more_details lora, more scale = more details, disabled on 0", default=0.5,
        ),
        detail_tweaker_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ), 
        film_grain_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ), 
        epi_noise_offset_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ),
        color_temprature_slider_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ),
        mp_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ),
        id_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ),
        ex_v1_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ),
        custom_model_url: str = Input(description="URL to a custom model file (optional)", default=None),
    ) -> List[Path]:
        if custom_model_url:
            self.gen = self._load_custom_model(custom_model_url)
        else:
            self.gen = Generator(
                sd_path="SG161222/Realistic_Vision_V6.0_B1_noVAE",
                vae_path="stabilityai/sd-vae-ft-mse",
                use_compel=True,
                load_controlnets={"lineart", "mlsd", "canny", "depth", "inpainting"},
                load_ip_adapter=True
            )

        outputs = self.gen.predict(
            prompt=prompt,
            lineart_image=lineart_image, lineart_conditioning_scale=lineart_conditioning_scale,
            depth_conditioning_scale=depth_conditioning_scale, depth_image=depth_image,
            mlsd_image=mlsd_image, mlsd_conditioning_scale=mlsd_conditioning_scale,
            canny_conditioning_scale=canny_conditioning_scale, canny_image=canny_image,
            inpainting_image=inpainting_image, mask_image=mask_image, inpainting_conditioning_scale=inpainting_conditioning_scale,
            num_outputs=num_outputs, max_width=max_width, max_height=max_height,
            scheduler=scheduler, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            seed=seed, eta=eta,
            negative_prompt=negative_prompt,
            guess_mode=guess_mode, disable_safety_check=disable_safety_check,
            sorted_controlnets=sorted_controlnets,
            ip_adapter_image=ip_adapter_image, ip_adapter_weight=ip_adapter_weight,
            img2img=img2img_image, img2img_strength=img2img_strength, ip_ckpt=ip_adapter_ckpt,
            text_for_auto_mask=positive_auto_mask_text.split(",") if positive_auto_mask_text else None,
            negative_text_for_auto_mask=negative_auto_mask_text.split(",") if negative_auto_mask_text else None,
            add_more_detail_lora_scale=add_more_detail_lora_scale, 
            detail_tweaker_lora_weight=detail_tweaker_lora_weight, 
            film_grain_lora_weight=film_grain_lora_weight, 
            epi_noise_offset_lora_weight=epi_noise_offset_lora_weight, 
            color_temprature_slider_lora_weight=color_temprature_slider_lora_weight,
            mp_lora_weight=mp_lora_weight, 
            id_lora_weight=id_lora_weight,
            ex_v1_lora_weight=ex_v1_lora_weight,
        )

        output_paths = []
        for i, output in enumerate(outputs):
            output_path = f"/tmp/output_{i}.png"
            output.images[0].save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths