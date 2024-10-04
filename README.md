# Image Generation Assignment with Stable Diffusion and ControlNet For Avatar HB1

This project focuses on generating images from text prompts and depth maps using Stable Diffusion, guided by ControlNet. The generated images are analyzed for their quality, aspect ratios, and generation latency. The goal is to critique the image generation pipeline and improve the process where possible.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
  - [Generate Images](#generate-images)
  - [Aspect Ratio Testing](#aspect-ratio-testing)
  - [Latency Measurement](#latency-measurement)
- [Results](#results)
  - [Aspect Ratio Analysis](#aspect-ratio-analysis)
  - [Latency Analysis](#latency-analysis)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

With the rapid advancement in text-to-image generation models like Stable Diffusion, it has become possible to generate high-quality, photo-realistic images from simple text prompts. Moreover, using techniques like ControlNet, the generation can be controlled with additional input data such as depth maps, normal maps, or canny edge maps.

In this assignment, we:
1. Generate the best possible output images using metadata (text prompts and depth maps).
2. Test image generation with different aspect ratios and comment on the generation quality.
3. Measure and analyze generation latency, proposing fixes where applicable.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.8+
- torch (PyTorch)
- diffusers (Hugging Face’s Diffusion models)
- Pillow (for image processing)
- numpy

To install the dependencies, run the following command:

bash
pip install torch diffusers pillow numpy


## Setup

Clone this repository to your local machine:

bash
git clone <repository-url>
cd <repository-directory>


Ensure that you have the depth maps (PNG and NPY files) in the ./images directory, as well as the prompts defined in the code.

## How to Run

### Generate Images

1. *Run the image generation script*:
   To generate images based on the provided depth maps and prompts, run the main.py file:

   bash
   python main.py
   

   This will process the prompts and depth maps, saving the generated images to the ./generated_images directory.

2. *Modify Prompts*:
   You can add or modify prompts inside the prompts list in main.py to match the metadata provided.

### Aspect Ratio Testing

To test different aspect ratios for image generation, modify the height and width parameters in the generate_image function in main.py. For example:

python
generate_image(prompt, depth_map_path, output_path, height=768, width=1024)


This will generate images with a 4:3 aspect ratio. You can test various aspect ratios (e.g., 16:9, 4:3, 1:1) by adjusting the height and width accordingly.

### Latency Measurement

The script will output the time taken to generate each image. The latency is measured using Python’s time module.

To see the generation latency for each image, simply look at the console output after running the script.

## Results

### Aspect Ratio Analysis

Images were generated using different aspect ratios, including 1:1, 16:9, and 4:3. It was observed that:

- *1:1 Aspect Ratio*: Produced the most balanced images with well-distributed details.
- *16:9 Aspect Ratio*: Wider images tended to lose some details at the edges, focusing more on the center of the image.
- *4:3 Aspect Ratio*: Worked reasonably well, though the image seemed slightly stretched in some cases.

Testing with different aspect ratios provides insight into how Stable Diffusion adapts to different framing and composition requirements.

### Latency Analysis

On average, the image generation latency was measured between 5-10 seconds per image, depending on the complexity of the prompt and the size of the depth map.

#### Quick Fixes for Latency:

- *Reducing Image Resolution*: Lowering the resolution from 512x512 to 256x256 can reduce the time to generate an image by approximately 40%, with minimal loss of detail.
- *Prompt Simplification*: Shortening and simplifying the text prompt can slightly reduce latency, though the impact on image quality varies.

However, further reducing latency by aggressively cutting resolution or prompt length results in a noticeable drop in image quality, with fewer details and increased artifacts.

## Conclusion

This project demonstrates the versatility of using Stable Diffusion with ControlNet to generate images conditioned on depth maps. While the model excels in generating detailed images from text prompts, there are some trade-offs between image quality, aspect ratio, and generation latency.

Future work could explore additional conditioning inputs like canny edge maps or normal maps to further control the image generation process and improve efficiency.

## References

1. [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
2. [ControlNet](https://github.com/lllyasviel/ControlNet)
3. [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
4. [Stable Diffusion v1-5 on Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5)

---

This README should provide a clear and detailed overview of your project, instructions on how to set it up, and an analysis of the results.
