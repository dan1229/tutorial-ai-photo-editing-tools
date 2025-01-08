"""
Image Enhancement using Stable Diffusion
Requirements:
- Python 3.7+
- torch
- diffusers
- PIL

Installation:
pip install torch diffusers Pillow
OR
pipenv install

Usage:
1. Single image:
   python tutorial.py

2. Directory of images:
   Modify main() to point to your input/output directories
"""

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import logging

# Set up basic logging to see what's happening
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def setup_stable_diffusion() -> StableDiffusionImg2ImgPipeline:
    """
    Initialize the Stable Diffusion pipeline with recommended settings.

    This function demonstrates how to:
    1. Check for CUDA (GPU) availability
    2. Load the model with appropriate settings
    3. Enable optimizations for better performance

    Returns:
        A configured StableDiffusionImg2ImgPipeline ready for use
    """
    # Check if we can use GPU (CUDA) or need to use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Add error handling for model loading
    try:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for demonstration
        )
    except Exception as e:
        logging.error(f"Failed to load Stable Diffusion model: {str(e)}")
        raise

    # Move the pipeline to our device (GPU or CPU)
    pipeline = pipeline.to(device)

    # Enable memory optimizations if using GPU
    if device == "cuda":
        pipeline.enable_model_cpu_offload()  # Helps with memory usage
        pipeline.enable_attention_slicing()  # Helps with memory usage

    return pipeline


def enhance_image(
    pipeline: StableDiffusionImg2ImgPipeline,
    input_path: str,
    output_path: str,
    prompt: str = "high quality, detailed image",
    strength: float = 0.5,
) -> None:
    """
    Enhance a single image using Stable Diffusion.

    Args:
        pipeline: The configured Stable Diffusion pipeline
        input_path: Path to the input image
        output_path: Where to save the enhanced image
        prompt: Text description of desired enhancements
        strength: How much to modify the image (0.0 to 1.0)
                 - 0.0 means no change
                 - 1.0 means maximum change

    Raises:
        Exception: If image processing fails
    """
    try:
        # Load and prepare the image
        input_image = Image.open(input_path).convert("RGB")
        logging.info(f"Loaded image: {input_path}")

        # Process the image
        # The pipeline takes our input image and prompt, then generates a new version
        result = pipeline(
            prompt=prompt,
            image=input_image,
            strength=strength,  # How much to modify the image
            guidance_scale=7.5,  # How closely to follow the prompt
            num_inference_steps=30,  # More steps = higher quality but slower
        ).images[0]

        # Save the result
        result.save(output_path)
        logging.info(f"Enhanced image saved to: {output_path}")

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise


def process_directory(
    input_dir: str,
    output_dir: str,
    prompt: str = "high quality, detailed image",
    strength: float = 0.5,
) -> None:
    """
    Process all images in a directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save enhanced images
        prompt: Text description of desired enhancements
        strength: Strength of the enhancement effect
    """
    # Add input validation
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the pipeline once and reuse it
    pipeline = setup_stable_diffusion()

    # Process each image in the directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"enhanced_{filename}")

            logging.info(f"Processing: {filename}")
            enhance_image(pipeline, input_path, output_path, prompt, strength)


def main() -> None:
    """
    Example usage of the image enhancement pipeline.

    This demonstrates how to:
    1. Process a single image
    2. Process all images in a directory
    """
    # Example 1: Process a single image
    pipeline = setup_stable_diffusion()
    enhance_image(
        pipeline=pipeline,
        input_path="example.jpg",
        output_path="enhanced_example.jpg",
        prompt="high quality, detailed photo with enhanced colors",
        strength=0.5,
    )

    # Example 2: Process all images in a directory
    process_directory(
        input_dir="input_images",
        output_dir="enhanced_images",
        prompt="high quality, detailed photo with enhanced colors",
        strength=0.5,
    )


if __name__ == "__main__":
    main()
