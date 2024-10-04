import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np
import os
import time

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained models (Stable Diffusion and ControlNet)
sd_model = "runwayml/stable-diffusion-v1-5"
controlnet_model = "lllyasviel/controlnet-depth"

# Initialize the pipeline with ControlNet
controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16).to(device)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(sd_model, controlnet=controlnet, torch_dtype=torch.float16).to(device)

# Function to load depth maps
def load_depth_map(filepath):
    if filepath.endswith(".png"):
        return Image.open(filepath).convert("L")
    elif filepath.endswith(".npy"):
        return np.load(filepath)
    else:
        raise ValueError("Unsupported file format")

# Generate an image from a prompt and depth map
def generate_image(prompt, depth_map_path, output_path, height=512, width=512, seed=12345):
    generator = torch.manual_seed(seed)
    
    # Load the depth map
    depth_map = load_depth_map(depth_map_path)
    
    # Convert depth map to the format needed by the pipeline
    if isinstance(depth_map, np.ndarray):
        depth_map = Image.fromarray(depth_map)

    # Generate image
    start_time = time.time()
    result = pipeline(prompt=prompt, image=depth_map, generator=generator, height=height, width=width)
    elapsed_time = time.time() - start_time

    # Save generated image
    result.images[0].save(output_path)
    
    return elapsed_time, result.images[0]

# Generate images for all prompts and depth maps
def process_images(prompts, depth_map_dir, output_dir):
    for i, prompt in enumerate(prompts):
        depth_map_path = os.path.join(depth_map_dir, f"{i+1}.png")
        output_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
        
        elapsed_time, image = generate_image(prompt, depth_map_path, output_path)
        print(f"Generated image {i+1} in {elapsed_time:.2f} seconds")

# Example usage
prompts = [
    "A futuristic city skyline at sunset",
    "A forest covered in snow with mountains in the background",
    # Add more prompts as per the metadata
]

depth_map_dir = "./images"  # Folder containing depth maps
output_dir = "./generated_images"  # Folder to save generated images

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate all images
process_images(prompts, depth_map_dir, output_dir)
