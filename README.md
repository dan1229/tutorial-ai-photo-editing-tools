# Tutorial - AI Photo Editing Tools

#### By [Daniel Nazarian](https://danielnazarian.com)

[![Black](https://github.com/dan1229/ai-photo-editor/actions/workflows/python-checks.yml/badge.svg?branch=main&event=push&job=black)](https://github.com/dan1229/ai-photo-editor/actions/workflows/python-checks.yml)
[![Flake8](https://github.com/dan1229/ai-photo-editor/actions/workflows/python-checks.yml/badge.svg?branch=main&event=push&job=flake8)](https://github.com/dan1229/ai-photo-editor/actions/workflows/python-checks.yml)
[![Types](https://github.com/dan1229/ai-photo-editor/actions/workflows/python-checks.yml/badge.svg?branch=main&event=push&job=mypy)](https://github.com/dan1229/ai-photo-editor/actions/workflows/python-checks.yml)

A Python-based tool that uses Stable Diffusion to enhance images. This tool can process entire directories of images, improving their quality, lighting, and details using various specialized presets. Perfect for batch processing photos for professional use, social media, or portfolio enhancement.

TODO - add link to blog post

## Features
- Batch process multiple images in a directory
- Multiple enhancement presets (Natural, Subtle, Maximum quality)
- Sample mode to preview all presets
- Graceful handling of interruptions (Ctrl+C)
- GPU acceleration (when available)
- Automatic output directory creation
- Detailed enhancement metadata logging

## Requirements
- Python 3.12 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/{owner}/{repo}.git
cd image-enhancement-tool
```

2. Install dependencies:
```bash
pipenv install
```

## Usage

Basic usage:
```bash
python main.py <input_directory> [output_directory]
```

Sample all presets:
```bash
python main.py <input_directory> --sample
```

Using specific preset:
```bash
python main.py <input_directory> --preset <preset_name>
```

### Available Presets and Sizes

The tool includes several presets optimized for different enhancement needs:
- **Default**: Balanced enhancement for general use
- **Subtle**: Minimal enhancement while preserving natural look
- **Natural**: Focus on realistic improvements
- **Maximum**: High-impact enhancement for professional results

Image sizes range from small (768px) to extra-large (1440px). See `presets.py` for full details.

### Examples

```bash
# Sample all presets (recommended for first use)
python main.py ./my_images --sample

# Use natural preset with large size
python main.py ./my_images --preset natural --size large

# Process photos with custom output directory
python main.py ./my_images -o ./enhanced_photos --preset default --size xl
```

### Output Structure

The tool creates organized output directories with the following structure:

```
out/enhanced_<input_dir>_<timestamp>/
    <preset>_<size>/
        enhancement_info.txt  # Contains all processing details
        [enhanced images]
```

When using --sample, it creates subdirectories for each preset:

```
out/enhanced_<input_dir>_<timestamp>/
    default_medium/
    subtle_medium/
    natural_medium/
    maximum_medium/
```

---

Copyright © 2024 [Daniel Nazarian](https://danielnazarian.com)