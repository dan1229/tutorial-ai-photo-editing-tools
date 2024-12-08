# Tutorial - AI Photo Editing Tools

[![Python Checks](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml/badge.svg?branch=main&event=push&job=black)](https://github.com/dan1229/tutorial-ai-photo-editing-tools/actions/workflows/python-checks.yml)

#### By [Daniel Nazarian](https://danielnazarian.com)


## Description

Learn how to use open source AI models to enhance photos in bulk using Python and Stable Diffusion. This tutorial demonstrates how to build a practical tool for automatically improving image quality, lighting, and details across entire directories of photos.

### What You'll Learn
- Using Stable Diffusion for photo enhancement
- Processing multiple images efficiently
- Working with enhancement presets
- Handling GPU acceleration
- Managing batch operations

## Getting Started

### Requirements
- Python 3.12 or higher
- CUDA-compatible GPU (recommended)
- 8GB RAM minimum
- Python packages listed in `Pipfile`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dan1229/tutorial-ai-photo-editing-tools.git
cd tutorial-ai-photo-editing-tools
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

## Usage

### 1. Basic Enhancement
Enhance all images in a directory:
```bash
python main.py ./my_images
```

### 2. Preset Options

Available enhancement presets:

- **Default**: Balanced, all-purpose enhancement
- **Subtle**: Light touch-ups
- **Natural**: Realistic improvements
- **Maximum**: Professional-grade enhancement

Try all presets:

```bash
python main.py ./my_images --sample
```

### 3. Custom Enhancement

Specify a preset and size:

```bash
python main.py ./my_images --preset natural --size large
```

### Output Structure

Enhanced images are organized as follows:

```
out/enhanced_<input_dir>_<timestamp>/
    <preset>_<size>/
        enhancement_info.txt
        [enhanced images]
```

When using --sample:

```
out/enhanced_<input_dir>_<timestamp>/
    default_medium/
    subtle_medium/
    natural_medium/
    maximum_medium/
```

---

Copyright © 2024 [Daniel Nazarian](https://danielnazarian.com)