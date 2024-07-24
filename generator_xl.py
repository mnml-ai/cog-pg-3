#@title Generator Class

#writing this class seperate from cog :: BECAUSE I HATE DOCKER :: And to make it work in colab and envs where docker is not available

#using t2i adapters instead of controlnets because xl controlnets results are not good

import torch
import os
from PIL import Image

from diffusers import (
    T2IAdapter, 
    AutoencoderKL, 
    StableDiffusionXLAdapterPipeline, 
    StableDiffusionXLInpaintPipeline,
    MultiAdapter,
    StableDiffusionXLPipeline,
    DiffusionPipeline
)
from compel import Compel, ReturnedEmbeddingsType
from diffusers.models import AutoencoderKL
from transformers import CLIPVisionModelWithProjection
# from utils import *

class GeneratorXL:
    def __init__(self, sd_path="stabilityai/stable-diffusion-xl-base-1.0", vae_path= "madebyollin/sdxl-vae-fp16-fix", load_ip_adapter=False, use_compel= True, ip_image_encoder= "weights/image_encoder", ip_weight="weights/ip-adapter_sd15.bin", load_t2i_adapters={} ):

        self.use_compel = use_compel
        self.load_t2i_adapters = load_t2i_adapters
        self.load_ip_adapter = load_ip_adapter
        self.ip_weight = ip_weight
        self.preprocessors = {}
        self.detectors = {}
        self.t2i_adapters= {}

        if vae_path:
            vae = AutoencoderKL.from_pretrained(vae_path)

        if load_ip_adapter:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(ip_image_encoder, local_files_only=True,).to("cuda", dtype=torch.float16)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            sd_path, torch_dtype=torch.float16,
            # local_files_only=True,
            vae= vae if vae_path else None
        ).to("cuda")

        if load_t2i_adapters:
            for name in load_t2i_adapters:
                print("loading t2i adapter...")
                model= AUX_IDS[name]
                self.t2i_adapters[name] = T2IAdapter.from_pretrained(
                    model["t2i_adapter_xl_path"], varient="fp16",
                    torch_dtype=torch.float16,
                    # local_files_only=True,
                ).to("cuda")
                print("loading control detectors..")
                self.detectors[name] = model['detector']()

        if self.use_compel:
            self.compel_proc =  Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2] , text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    def build_pipe(
            self, inputs, max_width, max_height, guess_mode=False, use_ip_adapter= False
        ):
        print("using ip adapter::", use_ip_adapter)
        if use_ip_adapter:
            from Diffusers_IPAdapter.ip_adapter.ip_adapter import IPAdapter
        t2i_adapters_ = []
        processed_control_images = []
        conditioning_scales = []
        w, h = max_width, max_height
        inpainting = False
        #image and mask for inpainting
        mask= None
        init_image= None
        got_size= False
        for name, [image, conditioning_scale, mask_image] in inputs.items():
            if image is None:
                continue
            # print(name)
            image = Image.open(image)
            if not got_size:
                image= resize_image(image, max_width, max_height)
                w, h= image.size
                got_size= True
            else:
                image= image.resize((w,h))

            if name=="inpainting" and mask_image:
                inpainting = True
                mask_image= Image.open(mask_image)
                mask= mask_image.resize((w,h))
                img= AUX_IDS[name]["preprocessor"](self, image, mask)
                init_image= image
            else:
                img= AUX_IDS[name]["preprocessor"](self, image)

            if not inpainting:
                img= img.resize((w,h))
            t2i_adapters_.append(self.t2i_adapters[name])
            processed_control_images.append(img)
            conditioning_scales.append(conditioning_scale)

        ip= None
        if len(t2i_adapters_)==0:
            pipe = self.pipe
            kwargs = {"width":max_width, "height": max_height}
            if use_ip_adapter:
                ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
        elif self.load_t2i_adapters:
            if len(t2i_adapters_)==1:
                adapter= t2i_adapters_[0]
                conditioning_scales= conditioning_scales[0]
            else:
                adapter= MultiAdapter(t2i_adapters_)

            if inpainting:
                # StableDiffusionXLControlNetAdapterInpaintPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    custom_pipeline="pipeline_stable_diffusion_xl_controlnet_adapter_inpaint",
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    adapter=adapter
                )
                kwargs = {
                    "image": init_image,
                    "mask_image": mask,
                    "adapter_image": processed_control_images,
                    "adapter_conditioning_scale": conditioning_scales,
                    "controlnet_conditioning_scale" : 0
                    # "guess_mode": guess_mode,
                }
            else:
                pipe = StableDiffusionXLAdapterPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    adapter= adapter,
                )

                if use_ip_adapter:
                    ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")

                kwargs = {
                    "image": processed_control_images,
                    "adapter_conditioning_scale": conditioning_scales,
                    # "guess_mode": guess_mode,
                }
        

        return pipe, kwargs, ip

    def predict(self, prompt="", lineart_image=None, lineart_conditioning_scale=1.0,
                scribble_image=None, scribble_conditioning_scale=1.0,
                tile_image=None, tile_conditioning_scale=1.0,
                brightness_image=None, brightness_conditioning_scale=1.0,
                inpainting_image=None, mask_image=None, inpainting_conditioning_scale=1.0,
                num_outputs=1, max_width=512, max_height=512,
                scheduler="DDIM", num_inference_steps=20, guidance_scale=7.0,
                seed=None, eta=0.0,
                negative_prompt="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                guess_mode=False, disable_safety_check=False,
                sorted_t2i_adapters="tile, inpainting, lineart",
                ip_adapter_image=None, ip_adapter_weight=1.0):
        
        if not disable_safety_check and 'nude' in prompt:
            raise Exception(
                f"NSFW content detected. try a different prompt."
            )
        #dont know why, if ip adapter image is not given, it produce green image- so quick fix for non-ip adapter generations - will it soon MAYBE
        if not ip_adapter_image:
            # ip_adapter_image= 'example/cat.png'
            ip_adapter_weight= 0.0

        control_inputs= {
                # "brightness": [brightness_image, brightness_conditioning_scale, None],
                # "tile": [tile_image, tile_conditioning_scale, None],
                "lineart": [lineart_image, lineart_conditioning_scale, None],
                "inpainting": [inpainting_image, inpainting_conditioning_scale, mask_image],
                "scribble": [scribble_image, scribble_conditioning_scale, None],
                # "depth": [depth_image, depth_conditioning_scale, None],
            }
        sorted_control_inputs= sort_dict_by_string(sorted_t2i_adapters, control_inputs)

        pipe, kwargs, ip = self.build_pipe(
            sorted_control_inputs,
            max_width=max_width,
            max_height=max_height,
            guess_mode=guess_mode,
            use_ip_adapter= ip_adapter_image
        )
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        pipe.to("cuda", torch.float16)
        # pipe.enable_xformers_memory_efficient_attention()

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # generator = torch.Generator("cuda").manual_seed(seed)

        if disable_safety_check:
            pipe.safety_checker = None

        outputs= []
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(seed)
            if ip_adapter_image:
                print("ip adapter---")
                ip_image= Image.open(ip_adapter_image)
                prompt_embeds_, negative_prompt_embeds_ = ip.get_prompt_embeds(
                    ip_image,
                    p_embeds=self.compel_proc(prompt),
                    n_embeds=self.compel_proc(negative_prompt),
                    weight=[ip_adapter_weight]
                )
                output = pipe(
                    prompt_embeds= prompt_embeds_,
                    negative_prompt_embeds= negative_prompt_embeds_,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=1,
                    generator=generator,
                    # output_type="pil",
                    **kwargs,
                )
            else:
                if self.use_compel:
                    conditioning, pooled = self.compel_proc(prompt)
                    n_c, n_p= self.compel_proc(negative_prompt)
                    kwargs["prompt_embeds"] = conditioning
                    kwargs["negative_prompt_embeds"] = n_c
                    kwargs["pooled_prompt_embeds"] = pooled
                    kwargs["negative_pooled_prompt_embeds"] = n_p
                else:
                    kwargs["prompt"] = prompt
                    kwargs["negative_prompt"] = negative_prompt
                output = pipe(
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=1,
                    generator=generator,
                    # output_type="pil",
                    **kwargs,
                )

            outputs.append(output)
        return outputs