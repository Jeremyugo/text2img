from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from stable_diffusion import run_stable_diffusion

# creating the app
app = FastAPI()

# defining the route for generating the image
@app.get("/text2img")
def run_text2img(prompt: str, *,
    seed: int = 1000, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    
    # runs the stable difussion function that generates the images
    image = run_stable_diffusion(prompt, seed=seed, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale)
    
    image.save("text2img_output.png")
    
    return FileResponse("text2img_output.png")
