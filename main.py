import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import signal
from datetime import datetime
import argparse
from typing import Optional, Any
from presets import PRESETS, SIZE_PRESETS
import logging
import numpy as np
from scipy import ndimage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Global flag to indicate interruption during processing
# This allows for graceful shutdown when Ctrl+C is pressed
interrupted: bool = False


def load_pipeline() -> StableDiffusionImg2ImgPipeline:
    """
    Initialize and configure the Stable Diffusion pipeline.

    Returns:
        StableDiffusionImg2ImgPipeline: Configured pipeline ready for image processing
    """
    try:
        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logging.info(f"Initializing pipeline with device: {device}, dtype: {dtype}")

        # Initialize pipeline with appropriate device settings
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None,
        )

        # Move to device and optimize
        pipeline = pipeline.to(device)
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_tiling()
            pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing("max")

        # Add scheduler configuration
        pipeline.scheduler.set_timesteps(50)

        # Verify pipeline components
        logging.info("Pipeline components loaded:")
        logging.info(f"- Text Encoder: {pipeline.text_encoder is not None}")
        logging.info(f"- VAE: {pipeline.vae is not None}")
        logging.info(f"- UNet: {pipeline.unet is not None}")

        logging.info(f"Pipeline configured successfully on {device}")
        return pipeline  # type: ignore[no-any-return]

    except Exception as e:
        logging.error(f"Error initializing pipeline: {str(e)}")
        logging.error(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.error(f"CUDA device: {torch.cuda.get_device_name()}")
            logging.error(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
            )
        raise


def process_image(
    input_image_path: str,
    output_image_path: str,
    pipeline: StableDiffusionImg2ImgPipeline,
    preset: str = "default",
    size: str = "medium",
) -> None:
    """
    Process a single image using the Stable Diffusion pipeline.

    Args:
        input_image_path: Path to the input image
        output_image_path: Where to save the enhanced image
        pipeline: Configured Stable Diffusion pipeline
        preset: Name of the enhancement preset to use
        size: Size preset for the output image
    """
    preset_config = PRESETS[preset]
    input_image: Optional[Image.Image] = None

    logging.info("=" * 100)

    # Step 1: Load and prepare image
    try:
        logging.info(f"Processing image: {input_image_path}")
        input_image = Image.open(input_image_path).convert("RGB")
        logging.info(f"Input image size: {input_image.size}, mode: {input_image.mode}")
    except Exception as e:
        logging.error(f"Error loading image {input_image_path}: {str(e)}")
        raise

    # Step 2: Resize image
    try:
        target_size = SIZE_PRESETS[size]
        width, height = input_image.size
        aspect_ratio: float = width / height

        if width > height:
            output_width = target_size
            output_height = int(target_size / aspect_ratio)
        else:
            output_height = target_size
            output_width = int(target_size * aspect_ratio)

        input_image = input_image.resize((output_width, output_height))
        logging.info(f"Resized to: ({output_width}, {output_height})")
    except Exception as e:
        logging.error(f"Error resizing image {input_image_path}: {str(e)}")
        raise

    # Step 3: Run pipeline with improved parameters
    try:
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Adjust the number of inference steps and eta for less graininess
        adjusted_steps = max(
            5, preset_config["num_inference_steps"] - 5  # type: ignore[operator]
        )  # Reduce steps
        adjusted_eta = 0.1  # Lower eta for less noise

        with torch.inference_mode():
            pipeline_result = pipeline(  # type: ignore[operator]
                prompt=preset_config["prompt"],
                negative_prompt=preset_config["negative_prompt"],
                image=input_image,
                strength=preset_config["strength"],
                guidance_scale=preset_config["guidance_scale"],
                num_inference_steps=adjusted_steps,  # Use adjusted steps
                eta=adjusted_eta,  # Use adjusted eta
                generator=torch.Generator().manual_seed(42),
            )

        # Add quality check
        if pipeline_result and hasattr(pipeline_result, "images"):
            result_image = pipeline_result.images[0]

            # Basic artifact detection
            if has_artifacts(result_image):
                logging.warning("Detected potential artifacts in generated image")
                # Optionally retry with different parameters or apply post-processing

        if pipeline_result is None:
            raise ValueError("Pipeline returned None")

        if not hasattr(pipeline_result, "images"):
            raise ValueError(
                f"Pipeline result missing 'images' attribute: {type(pipeline_result)}"
            )

        result_image = pipeline_result.images[0]

    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA out of memory - try reducing image size or using CPU")
        raise
    except Exception as e:
        logging.error(f"Error during pipeline inference: {str(e)}")
        logging.error(f"Preset config: {preset_config}")
        raise

    # Step 4: Extract and save result
    try:
        result_image.save(output_image_path)
        logging.info(f"Enhanced image saved to {output_image_path}")
    except Exception as e:
        logging.error(f"Error saving result: {str(e)}")
        logging.error(f"Pipeline result type: {type(pipeline_result)}")
        if hasattr(pipeline_result, "images"):
            logging.error(f"Images attribute type: {type(pipeline_result.images)}")
        raise
    finally:
        if input_image is not None:
            del input_image


def has_artifacts(image: Image.Image) -> bool:
    """
    Basic artifact detection function.
    Returns True if potential artifacts are detected.
    """
    # Convert to numpy array for analysis
    img_array = np.array(image)

    # Check for extreme color variations
    local_std = np.std(img_array, axis=(0, 1))
    if np.any(local_std > 100):  # Threshold for color variation
        return True

    # Check for unnatural edges
    edges = ndimage.sobel(img_array.mean(axis=2))
    if np.percentile(np.abs(edges), 99) > 200:  # Threshold for edge detection
        return True

    return False


def process_directory(
    input_directory: str,
    output_directory: str,
    preset: str = "default",
    size: str = "medium",
    total_presets: int = 1,
    current_preset_num: int = 1,
) -> None:
    global interrupted

    if not os.path.exists(input_directory):
        logging.error(f"Error: The directory {input_directory} does not exist.")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Count total number of images first
    total_images = sum(
        1
        for f in os.listdir(input_directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if total_images == 0:
        logging.info("No images found in directory.")
        return

    pipeline: StableDiffusionImg2ImgPipeline = load_pipeline()

    def signal_handler(sig: int, frame: Any) -> None:
        global interrupted
        logging.info("\nProcess interrupted. Cleaning up...")
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)

    processed = 0
    for filename in os.listdir(input_directory):
        if interrupted:
            logging.info("Exiting due to interruption.")
            break

        input_path = os.path.join(input_directory, filename)

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            processed += 1
            output_path = os.path.join(output_directory, filename)

            # First log the processing message
            logging.info(f"Processing image: {filename}")

            # Then log the detailed progress
            if total_presets > 1:
                total_to_process = total_images * total_presets
                overall_progress = (total_images * (current_preset_num - 1)) + processed
                logging.info(f"Progress: {processed}/{total_images} images in preset")
                logging.info(f"\tPreset: {current_preset_num}/{total_presets}")
                logging.info(f"\tOverall: {overall_progress}/{total_to_process}")
            else:
                logging.info(f"Progress: {processed}/{total_images} images")

            try:
                process_image(input_path, output_path, pipeline, preset, size)
            except KeyboardInterrupt:
                logging.info("Exiting due to interruption.")
                break
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
        else:
            logging.info(f"Skipping non-image file: {filename}")

    logging.info(f"Completed processing {processed}/{total_images} images")
    del pipeline


def write_enhancement_metadata(
    output_directory: str, preset_name: str, size: str, sample: bool = False
) -> None:
    """Write metadata about the enhancement process to a file."""
    preset_config = PRESETS[preset_name]

    metadata = f"""Enhancement performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
SAMPLE: {sample}
Preset: {preset_config['name']} ({preset_name})
Model: Stable Diffusion 1.5
Size: {size} ({SIZE_PRESETS[size]}px)
Prompt: {preset_config['prompt']}
Negative prompt: {preset_config['negative_prompt']}
Strength: {preset_config['strength']}
Guidance scale: {preset_config['guidance_scale']}
Steps: {preset_config['num_inference_steps']}"""

    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, "enhancement_info.txt"), "w") as f:
        f.write(metadata)


def main() -> None:
    """
    Main entry point for the image enhancement tool.
    Handles argument parsing and orchestrates the enhancement process.
    """
    parser = argparse.ArgumentParser(
        description="Enhance images using Stable Diffusion"
    )
    parser.add_argument("input_directory", help="Directory containing input images")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Optional custom output directory "
            "(default: enhanced_<input_dir>_<timestamp>)"
        ),
        default=None,
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run all presets on the input directory",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="default",
        help=f"Preset configuration to use (default: default). "
        f"Available presets: {', '.join(PRESETS.keys())}",
    )
    parser.add_argument(
        "--size",
        choices=list(SIZE_PRESETS.keys()),
        default="medium",
        help=f"Size of the output image (default: medium). "
        f"Available sizes: {', '.join(SIZE_PRESETS.keys())}",
    )

    args: argparse.Namespace = parser.parse_args()

    # Set up base output directory with improved structure
    if args.output:
        base_output_dir = args.output
    else:
        # Format: 2024-03-15_14-30-45
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        input_dir_name = os.path.basename(os.path.normpath(args.input_directory))
        base_output_dir = os.path.join(
            "out",
            input_dir_name,  # Group by input directory name
            timestamp,  # Then by timestamp
        )

    # Print startup summary
    logging.info("\n=== Image Enhancement Tool ===")
    logging.info(f"Input directory: {args.input_directory}")
    logging.info(f"Mode: {'Sample (all presets)' if args.sample else 'Single preset'}")
    if not args.sample:
        logging.info(f"Selected preset: {args.preset}")
    logging.info(f"Output size: {args.size} ({SIZE_PRESETS[args.size]}px)")
    logging.info(f"Output directory: {os.path.abspath(base_output_dir)}")
    logging.info("===========================\n")

    if args.sample:
        # Process all presets
        total_presets = len(PRESETS)
        for i, preset_name in enumerate(PRESETS.keys(), 1):
            output_directory = os.path.join(
                base_output_dir,
                f"{preset_name}_{args.size}",
            )
            write_enhancement_metadata(output_directory, preset_name, args.size, True)
            process_directory(
                args.input_directory,
                output_directory,
                preset_name,
                args.size,
                total_presets=total_presets,
                current_preset_num=i,
            )
            if interrupted:
                break
    else:
        # Single preset processing
        output_directory = os.path.join(
            base_output_dir,
            f"{args.preset}_{args.size}",
        )
        write_enhancement_metadata(output_directory, args.preset, args.size, False)
        process_directory(
            args.input_directory, output_directory, args.preset, args.size
        )


if __name__ == "__main__":
    main()
