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

## Output Structure

```
out/enhanced_<input_dir>_<timestamp>/
    <preset>_<size>/
        enhancement_info.txt
        [enhanced images]
```



## Customization

Feel free to fork this repository and adapt it to your specific needs! The codebase is designed to be modular and extensible. You can easily add new presets, modify enhancement parameters, or integrate additional AI models to suit your project requirements.

---
