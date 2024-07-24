import torch
import os
import requests
from urllib.parse import urlparse, unquote
from safetensors.torch import load_file
from PIL import Image, ImageEnhance
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    AutoencoderKL
)
from compel import Compel
from transformers import CLIPVisionModelWithProjection
from utils import *
import time
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from torch import nn
import numpy as np
import logging

# Constants
MODEL_CACHE = "model_cache"
MAX_CACHED_MODELS = 3
DEVICE = "cuda"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator:
    def __init__(self):
        self.model_pipes = {}
        self.controlnets = {}
        self.detectors = {}
        os.makedirs(MODEL_CACHE, exist_ok=True)
        self.setup_controlnet_detectors()

    def setup_controlnet_detectors(self):
        from controlnet_aux import OpenposeDetector, MidasDetector, HEDdetector, MLSDdetector, CannyDetector, LineartDetector
        self.controlnet_detectors = {
            "depth": MidasDetector.from_pretrained("lllyasviel/ControlNet"),
            "openpose": OpenposeDetector.from_pretrained("lllyasviel/ControlNet"),
            "hed": HEDdetector.from_pretrained("lllyasviel/ControlNet"),
            "mlsd": MLSDdetector.from_pretrained("lllyasviel/ControlNet"),
            "canny": CannyDetector.from_pretrained("lllyasviel/ControlNet"),
            "lineart": LineartDetector.from_pretrained("lllyasviel/ControlNet"),
        }

    def load_model(self, model_name: str = "SG161222/Realistic_Vision_V6.0_B1_noVAE", model_url: str = None):
        model_key = model_url or model_name
        if model_key in self.model_pipes:
            logger.info(f"Using cached model: {model_key}")
            self.model_pipes[model_key]['last_used'] = time.time()
            return self.model_pipes[model_key]['pipe']

        self._manage_model_cache()

        if model_url:
            pipe = self._load_custom_model(model_url)
        else:
            pipe = self._load_predefined_model(model_name)

        self._setup_pipeline(pipe)
        self._init_compel(pipe)
        self.model_pipes[model_key] = {'pipe': pipe, 'last_used': time.time()}
        
        return pipe

    def _init_compel(self, pipe):
        self.compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    def _manage_model_cache(self):
        if len(self.model_pipes) >= MAX_CACHED_MODELS:
            lru_model = min(self.model_pipes, key=lambda k: self.model_pipes[k]['last_used'])
            logger.info(f"Removing least recently used model from cache: {lru_model}")
            del self.model_pipes[lru_model]

    def _load_custom_model(self, model_url: str):
        model_path = self._download_model(model_url)
        logger.info(f"Loading custom model from {model_path}")
        try:
            return StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to(DEVICE)
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise ValueError(f"Failed to load custom model from {model_url}")

    def _load_predefined_model(self, model_name: str):
        logger.info(f"Loading {model_name} pipeline...")
        try:
            return StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                cache_dir=MODEL_CACHE,
                revision="main",
            ).to(DEVICE)
        except Exception as e:
            logger.warning(f"Failed to load with safetensors: {e}")
            logger.info("Attempting to load with PyTorch weights...")
            return StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=False,
                cache_dir=MODEL_CACHE,
                revision="main",
            ).to(DEVICE)

    def _setup_pipeline(self, pipe):
        if pipe.vae is None:
            logger.info("VAE not found in the pipeline. Loading a default VAE.")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16,
                cache_dir=MODEL_CACHE
            ).to(DEVICE)
            pipe.vae = vae
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled successfully")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
            logger.info("Falling back to default attention mechanism")
        
        pipe.enable_model_cpu_offload()

    def _download_model(self, url: str) -> str:
        local_filename = os.path.join(MODEL_CACHE, os.path.basename(urlparse(url).path))
        
        if os.path.exists(local_filename):
            logger.info(f"Model file already exists: {local_filename}")
            return local_filename

        logger.info(f"Downloading model from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return local_filename

    def convert_image(self, image):
        grayscale_image = image.convert("L")
        new_image = Image.new("RGBA", grayscale_image.size)

        for x in range(grayscale_image.width):
            for y in range(grayscale_image.height):
                intensity = grayscale_image.getpixel((x, y))
                if intensity == 255:
                    new_image.putpixel((x, y), (0, 0, 0, 255))
                elif intensity == 0:
                    new_image.putpixel((x, y), (0, 0, 0, 0))

        return new_image

    def segment_image(self, texts, image, negative=False):
        if len(texts) == 1:
            texts = [texts[0], texts[0]]

        images = [image] * len(texts)

        inputs = self.clip_seg_processor(text=texts, images=images, padding=True, return_tensors="pt")
        outputs = self.clip_seg_model(**inputs)

        preds = nn.functional.interpolate(
            outputs.logits.unsqueeze(1),
            size=(images[0].size[1], images[0].size[0]),
            mode="bilinear"
        )

        blended_image = Image.fromarray((torch.sigmoid(preds[0][0]).detach().numpy() * 255).astype(np.uint8))

        for pred in preds[1:]:
            current_image = Image.fromarray((torch.sigmoid(pred[0]).detach().numpy() * 255).astype(np.uint8))
            blended_image = Image.blend(blended_image, current_image, alpha=0.5)

        enhancer = ImageEnhance.Contrast(blended_image)
        blended_image = enhancer.enhance(2.0)

        enhancer = ImageEnhance.Brightness(blended_image)
        blended_image = enhancer.enhance(2.5)
        
        if negative:
            blended_image = self.convert_image(blended_image)

        return blended_image

    def build_pipe(self, inputs, max_width, max_height, guess_mode=False, use_ip_adapter=False, img2img=None, img2img_strength=0.8):
        logger.info(f"Using IP adapter: {use_ip_adapter}")
        if use_ip_adapter:
            from Diffusers_IPAdapter.ip_adapter.ip_adapter import IPAdapter
        control_nets = []
        processed_control_images = []
        conditioning_scales = []
        w, h = max_width, max_height
        inpainting = False
        mask = None
        init_image = None
        got_size = False
        img2img_image = None
        for name, [image, conditioning_scale, mask_image, text_for_auto_mask, negative_text_for_auto_mask] in inputs.items():
            if image is None:
                continue
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            if not got_size:
                image = resize_image(image, max_width, max_height)
                w, h = image.size
                got_size = True
            else:
                image = image.resize((w,h))

            if name == "inpainting" and (mask_image or text_for_auto_mask or negative_text_for_auto_mask):
                inpainting = True
                if text_for_auto_mask:
                    logger.info("Generating mask")
                    ti = time.time()
                    mask = self.segment_image(text_for_auto_mask, image).resize((w,h))
                    logger.info(f"Time taken to generate mask: {time.time() - ti:.2f} seconds")
                    if negative_text_for_auto_mask:
                        ti = time.time()
                        n_mask = self.segment_image(negative_text_for_auto_mask, image, negative=True).resize((w,h))
                        mask = Image.alpha_composite(mask.convert("RGBA"), n_mask)
                        logger.info(f"Time taken to generate negative mask: {time.time() - ti:.2f} seconds")
                    logger.info(f"Image size: {image.size}, Mask size: {mask.size}")
                    img = AUX_IDS[name]["preprocessor"](self, image, mask)
                else:
                    mask_image = Image.open(mask_image)
                    mask = mask_image.resize((w,h))
                    img = AUX_IDS[name]["preprocessor"](self, image, mask)
                init_image = image
                inpaint_strength = conditioning_scale
                inpaint_img = img
            else:
                img = AUX_IDS[name]["preprocessor"](self, image)
                img = img.resize((w,h))

            control_nets.append(self.controlnets[name])
            processed_control_images.append(img)
            conditioning_scales.append(conditioning_scale)

        if img2img:
            logger.info(f'Image to image: {img2img}')
            if not isinstance(img2img, Image.Image):
                img2img_image = Image.open(img2img)
            if not got_size:
                logger.info("Size not set, resizing image")
                img2img_image = resize_image(img2img_image, max_width, max_height)
            else:
                try:
                    img2img_image = img2img_image.resize((w,h))
                except:
                    logger.warning("Error in resizing, falling back to resize_image function")
                    img2img_image = resize_image(img2img_image, max_width, max_height)

        ip = None
        if len(control_nets) == 0:
            pipe = self.pipe
            kwargs = {"width": max_width, "height": max_height}
            if use_ip_adapter:
                ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
        else:
            if inpainting:
                pipe = StableDiffusionControlNetInpaintPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=self.pipe.safety_checker,
                    feature_extractor=self.pipe.feature_extractor,
                    controlnet=control_nets,
                )
                if use_ip_adapter:
                    ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
                kwargs = {
                    "image": init_image,
                    "mask_image": mask,
                    "control_image": processed_control_images,
                    "controlnet_conditioning_scale": conditioning_scales,
                    "guess_mode": guess_mode,
                    "strength": inpaint_strength
                }
            elif img2img:
                pipe = StableDiffusionControlNetImg2ImgPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=self.pipe.safety_checker,
                    feature_extractor=self.pipe.feature_extractor,
                    controlnet=control_nets,
                )
                if use_ip_adapter:
                    ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
                kwargs = {
                    "image": img2img_image,
                    "control_image": processed_control_images,
                    "controlnet_conditioning_scale": conditioning_scales,
                    "guess_mode": guess_mode,
                    "strength": img2img_strength
                }
            else:
                pipe = StableDiffusionControlNetPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=self.pipe.safety_checker,
                    feature_extractor=self.pipe.feature_extractor,
                    controlnet=control_nets,
                )
                if use_ip_adapter:
                    ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
                kwargs = {
                    "image": processed_control_images,
                    "controlnet_conditioning_scale": conditioning_scales,
                    "guess_mode": guess_mode,
                }
        return pipe, kwargs, ip

    def predict(self, prompt="", lineart_image=None, lineart_conditioning_scale=1.0,
                scribble_image=None, scribble_conditioning_scale=1.0,
                tile_image=None, tile_conditioning_scale=1.0,
                brightness_image=None, brightness_conditioning_scale=1.0,
                inpainting_image=None, mask_image=None, inpainting_conditioning_scale=1.0,
                depth_conditioning_scale=1.0, depth_image=None,
                mlsd_image=None, mlsd_conditioning_scale=1.0,
                canny_conditioning_scale=1.0, canny_image=None,
                num_outputs=1, max_width=512, max_height=512,
                scheduler="DDIM", num_inference_steps=20, guidance_scale=7.0,
                seed=None, eta=0.0,
                negative_prompt="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                guess_mode=False, disable_safety_check=False,
                sorted_controlnets="tile, inpainting, lineart",
                ip_adapter_image=None, ip_adapter_weight=1.0,
                img2img=None, img2img_strength=0.8, ip_ckpt='"ip-adapter_sd15.bin"',
                text_for_auto_mask=None, negative_text_for_auto_mask=None,
                add_more_detail_lora_scale=0, detail_tweaker_lora_weight=0, film_grain_lora_weight=0, 
                epi_noise_offset_lora_weight=0, color_temprature_slider_lora_weight=0,
                mp_lora_weight=0, id_lora_weight=0, ex_v1_lora_weight=0):
        
        lora_weights = []
        loras = []
        if add_more_detail_lora_scale != 0:
            lora_weights.append(add_more_detail_lora_scale)
            loras.append("more_details")
        if detail_tweaker_lora_weight != 0:
            lora_weights.append(detail_tweaker_lora_weight)
            loras.append("add_detail")
        if film_grain_lora_weight != 0:
            lora_weights.append(film_grain_lora_weight)
            loras.append("FilmVelvia3")
        if epi_noise_offset_lora_weight != 0:
            lora_weights.append(epi_noise_offset_lora_weight)
            loras.append("epi_noiseoffset2")
        if color_temprature_slider_lora_weight != 0:
            lora_weights.append(color_temprature_slider_lora_weight)
            loras.append("color_temperature_slider_v1")
        if mp_lora_weight != 0:
            lora_weights.append(mp_lora_weight)
            loras.append("mp_v1")
        if id_lora_weight != 0:
            lora_weights.append(id_lora_weight)
            loras.append("id_v1")
        if ex_v1_lora_weight != 0:
            lora_weights.append(ex_v1_lora_weight)
            loras.append("ex_v1")

        t1 = time.time()
        self.ip_weight = f"weights/{ip_ckpt}"

        if not disable_safety_check and 'nude' in prompt:
            raise Exception("NSFW content detected. Try a different prompt.")

        if not ip_adapter_image:
            ip_adapter_image = 'example/cat.png'
            ip_adapter_weight = 0.0

        control_inputs = {
            "brightness": [brightness_image, brightness_conditioning_scale, None, None, None],
            "tile": [tile_image, tile_conditioning_scale, None, None, None],
            "lineart": [lineart_image, lineart_conditioning_scale, None, None, None],
            "inpainting": [inpainting_image, inpainting_conditioning_scale, mask_image, text_for_auto_mask, negative_text_for_auto_mask],
            "scribble": [scribble_image, scribble_conditioning_scale, None, None, None],
            "depth": [depth_image, depth_conditioning_scale, None, None, None],
            "mlsd": [mlsd_image, mlsd_conditioning_scale, None, None, None],
            "canny": [canny_image, canny_conditioning_scale, None, None, None],
        }
        sorted_control_inputs = sort_dict_by_string(sorted_controlnets, control_inputs)
        t2 = time.time()
        logger.info(f"Time taken until build pipe: {t2 - t1:.2f} seconds")
        pipe, kwargs, ip = self.build_pipe(
            sorted_control_inputs,
            max_width=max_width,
            max_height=max_height,
            guess_mode=guess_mode,
            use_ip_adapter=ip_adapter_image,
            img2img=img2img, 
            img2img_strength=img2img_strength
        )
        t3 = time.time()
        logger.info(f"Time taken to build pipe: {t3 - t2:.2f} seconds")
        if scheduler == 'DPMSolverMultistep':
            pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
        else:
            pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        t4 = time.time()
        logger.info(f"Time taken to apply scheduler: {t4 - t3:.2f} seconds")
        t5 = time.time()
        logger.info(f"Time taken to cuda: {t5 - t4:.2f} seconds")
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        logger.info(f"Using seed: {seed}")

        if disable_safety_check:
            pipe.safety_checker = None

        outputs = []
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(seed)
            
            # Apply LoRA weights
            for lora, weight in zip(loras, lora_weights):
                pipe.load_lora_weights(f"dsgnrai/lora", weight_name=f"{lora}.safetensors", adapter_name=lora)
                pipe.set_adapters([lora], adapter_weights=[weight])
            
            if ip_adapter_image:
                t6 = time.time()
                logger.info(f"Time taken until ip: {t6 - t5:.2f} seconds")
                ip_image = Image.open(ip_adapter_image)
                prompt_embeds_, negative_prompt_embeds_ = ip.get_prompt_embeds(
                    ip_image,
                    p_embeds=self.compel_proc(prompt),
                    n_embeds=self.compel_proc(negative_prompt),
                    weight=[ip_adapter_weight]
                )
                t7 = time.time()
                logger.info(f"Time taken to load ip: {t7 - t6:.2f} seconds")
                output = pipe(
                    prompt_embeds=prompt_embeds_,
                    negative_prompt_embeds=negative_prompt_embeds_,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=1,
                    generator=generator,
                    **kwargs,
                )
                t8 = time.time()
                logger.info(f"Time taken to generate image: {t8 - t7:.2f} seconds")
            else:
                output = pipe(
                    prompt_embeds=self.compel_proc(prompt),
                    negative_prompt_embeds=self.compel_proc(negative_prompt),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=1,
                    generator=generator,
                    **kwargs,
                )
            
            # Unload LoRA weights
            pipe.unload_lora_weights()
            
            if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                continue
            outputs.append(output)
        t9 = time.time()
        return outputs