import os
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
token = os.environ.get('AUTH_TOKEN')

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                         use_auth_token=token, torch_dtype=torch.float16, use_safetensors=True, 
                                         variant="fp16")

pipe = pipe.to("cuda")


def run_stable_diffusion(
    prompt: str,
    *,
    seed: int = 1000,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Image:
    generator = (
        None if seed is None else torch.Generator(device="cuda").manual_seed(seed)
    )

    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return image
