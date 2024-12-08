# Tutorial - AI Photo Editing Tools

#### By [Daniel Nazarian](https://danielnazarian.com)

[![Black](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml/badge.svg?branch=main&event=push&job=black)](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml)
[![Flake8](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml/badge.svg?branch=main&event=push&job=flake8)](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml)
[![Types](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml/badge.svg?branch=main&event=push&job=mypy)](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml)


## Description

**TODO - add link to blog post**

Learn how to use open source AI models to enhance photos in bulk using Python and Stable Diffusion. This tutorial demonstrates how to build a practical tool for automatically improving image quality, lighting, and details across entire directories of photos.

### What You'll Learn
- How to use Stable Diffusion for photo enhancement
- Processing multiple images efficiently
- Working with different enhancement presets
- Handling GPU acceleration
- Managing large batch operations

## Getting Started

### Requirements
- Python 3.12 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- Required Python packages (see `requirements.txt`)



### Installation

1. Clone the tutorial repository:
```bash
git clone https://github.com/{owner}/{repo}.git
cd image-enhancement-tool
```

2. Set up your environment:
```bash
pipenv install
```

## How It Works

This tool leverages Stable Diffusion to enhance photos using various presets. Here's what happens under the hood:
1. Images are loaded and preprocessed
2. The AI model applies enhancement based on selected preset
3. Enhanced images are saved with metadata

## Step-by-Step Usage

### 1. Basic Enhancement
Start with a simple enhancement of all images in a directory:
```bash
python main.py ./my_images
```
tutorial-ai-photo-editing-tools


### 2. Try Different Presets
The tool includes several enhancement styles:
- **Default**: Balanced, all-purpose enhancement
- **Subtle**: Light touch-ups
- **Natural**: Realistic improvements
- **Maximum**: Professional-grade enhancement

Sample all presets to see what works best:
```bash
python main.py ./my_images --sample
```

### 3. Customize Your Enhancement
Choose a specific preset and size:
```bash
python main.py ./my_images --preset natural --size large
```

### Available Presets and Sizes

The tool includes several presets optimized for different enhancement needs:
- **Default**: Balanced enhancement for general use
- **Subtle**: Minimal enhancement while preserving natural look
- **Natural**: Focus on realistic improvements
- **Maximum**: High-impact enhancement for professional results

Image sizes range from small (768px) to extra-large (1440px). See `presets.py` for full details.


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