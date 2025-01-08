# AI Photo Enhancement Tools

[![Python Checks](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml/badge.svg?branch=main&event=push&job=black)](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml)

A Python-based tool for bulk photo enhancement using Stable Diffusion. Automatically improve image quality, lighting, and details across entire directories of photos.

## Features

- 🖼️ Batch photo enhancement using Stable Diffusion
- 🎨 Multiple enhancement presets
- 🚀 GPU acceleration support
- 📁 Directory-based processing
- 💾 Organized output structure

## Requirements

- Python 3.12+
- CUDA-compatible GPU (recommended)
- 8GB RAM minimum
- Dependencies in `Pipfile`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dan1229/tutorial-ai-photo-editing-tools.git
cd tutorial-ai-photo-editing-tools
```

2. Install dependencies:
```bash
pipenv install
```

## Usage

### Basic Enhancement

Run the script with the directory of images you want to enhance.

```bash
python main.py ./my_images
```

### Enhancement Presets

- **Default**: Balanced enhancement
- **Subtle**: Minimal adjustments
- **Natural**: Realistic improvements
- **Maximum**: High-impact enhancement

Try all presets:
```bash
python main.py ./my_images --sample
```

### Custom Options

```bash
python main.py ./my_images --preset natural --size large
```

### Output Structure

```
out/enhanced_<input_dir>_<timestamp>/
    <preset>_<size>/
        enhancement_info.txt
        [enhanced images]
```

## Tutorial Version

For those new to AI image enhancement, we provide a simplified `tutorial.py` that demonstrates the core concepts:

### Features
- 📚 Extensively documented code explaining each step
- 🔰 Simplified implementation focused on learning
- 🛠️ Basic image enhancement functionality
- 📖 Clear examples of single image and batch processing

### Usage

```bash
# Process a single image
python tutorial.py

# Or import and use the functions directly:
from tutorial import enhance_image, process_directory

# Process a directory of images
process_directory(
    input_dir="my_images",
    output_dir="enhanced",
    prompt="high quality photo with vivid colors",
    strength=0.5
)
```

### Key Parameters

- `prompt`: Text description of desired enhancements
- `strength`: Enhancement intensity (0.0 to 1.0)
  - 0.0: No change
  - 1.0: Maximum modification
- `guidance_scale`: How closely to follow the prompt (default: 7.5)
- `num_inference_steps`: Quality vs. speed tradeoff (default: 30)

## Customization

Feel free to fork this repository and adapt it to your specific needs! The codebase is designed to be modular and extensible. You can easily add new presets, modify enhancement parameters, or integrate additional AI models to suit your project requirements.

---
