# Define the base prompt to be used across presets
BASE_PROMPT = "enhance portrait quality, improve lighting and clarity"

# Define enhancement presets with carefully tuned parameters
PRESETS = {
    "default": {
        "name": "Smooth Enhancement",
        "prompt": f"{BASE_PROMPT} High-quality portrait with natural skin tones, vibrant colors, "
        "and subtle professional enhancements.",
        "negative_prompt": "deformed, distorted, disfigured, bad anatomy, "
        "unnatural, artifacts, blurry edges, oversaturated",
        "strength": 0.1,
        "guidance_scale": 50,
        "num_inference_steps": 10,
    },
    "subtle": {
        "name": "Subtle Enhancement",
        "prompt": f"{BASE_PROMPT} Professional photography, natural enhancement, preserve "
        "authentic features, subtle improvement in clarity and color, professional lighting",
        "negative_prompt": "artificial, fake, unrealistic, cartoon, anime, illustration, "
        "3d render, distorted, bad anatomy, oversaturated, artifacts, noise",
        "strength": 0.075,
        "guidance_scale": 50,
        "num_inference_steps": 5,
    },
    "natural": {
        "name": "Natural Enhancement",
        "prompt": f"{BASE_PROMPT} Professional portrait, subtle enhancement, natural texture, "
        "realistic lighting, photorealistic quality.",
        "negative_prompt": "artificial, fake, unrealistic, cartoon, overprocessed, "
        "artifacts, noise, grain, blurry edges",
        "strength": 0.075,
        "guidance_scale": 50,
        "num_inference_steps": 15,
    },
    "maximum": {
        "name": "Maximum Enhancement",
        "prompt": f"{BASE_PROMPT} professional studio quality with enhanced details. "
        "Ultra high-end professional portrait, masterful studio lighting, "
        "crisp details, magazine cover quality, photorealistic.",
        "negative_prompt": "blurry, noise, grain, low quality, artificial, oversaturated, "
        "artifacts, unnatural edges, distortion",
        "strength": 0.15,
        "guidance_scale": 50,
        "num_inference_steps": 5,
    },
}

# Define standard size presets that maintain aspect ratio
# The values represent the longest side of the image in pixels
SIZE_PRESETS = {
    "small": 768,  # Good for quick previews and web use
    "medium": 1024,  # Default - balanced size for most uses
    "large": 1280,  # High quality, larger file size
    "xl": 1440,  # Maximum quality, significant processing time
}
