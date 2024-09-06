import torch
from constants import DEVICE
import bentoml
from loguru import logger
from PIL.Image import Image
from diffusers import DiffusionPipeline

@bentoml.service(
    resources={"gpu": 1},
    traffic={
        "timeout":30, 
    },
)
class DiffusionAPI:
    def __init__(self) -> None:
        self.pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_safetensors=True,
        ).to(DEVICE)
        logger.info("Model Loaded")
        self.pipeline.enable_vae_slicing()
        logger.info("VAE Slicing Enabled")
        self.pipeline.enable_xformers_memory_efficient_attention()
        logger.info("Efficient Attention Enabled")

    @bentoml.api(route="/generate")
    def generate(
        self,
        text: str,
        num_inference_steps: int = 20
    ) -> Image:
        logger.info(f"Diffusion Starting, Text Received is {text}")
        image = self.pipeline(
            [text], num_inference_steps=num_inference_steps).images[0]
        logger.info(f"Image Generated, Sending")
        return image    