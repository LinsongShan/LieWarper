# LieWarper: Geometry-Aware Motion Transfer via Lie Algebra

## [Visualization](https://anonymous.4open.science/r/Anonymous-repository-of-Liewarper/DAVIS2017_Results.md)

## ⚙️ Environment Setup

This project has been tested on a **single NVIDIA A100 (40GB)** GPU.

### Installation

```
# 1. Create environment
conda create -n liewarper python=3.11 
conda activate liewarper

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 🚀 Inference

Place your reference videos in the `assets/` directory and run the following commands to start inference:

```python
python main.py \
     --input_video assets/motorbike.mp4 \
     --prompt "A wild horse gallops along a winding mountain road." \
     --frames 49
```

```python
python main.py \
     --input_video assets/drift-turn.mp4 \
     --prompt "A go-kart drifts around a corner on a race track." \
     --frames 49
```

```python
python main.py \
     --input_video assets/lady-running.mp4 \
     --prompt "A golden retriever runs hurriedly through a messy living room." \
     --frames 49
```

### Arguments

| **Argument**    | **Description**                                              |
| --------------- | ------------------------------------------------------------ |
| `--input_video` | Path to the reference video, used for extracting motion priors. |
| `--prompt`      | The text description for the target video.                   |
| `--frames`      | Number of generated frames (supports **49** or **81**).      |
| `--output_dir`  | (Optional) Path to save results. Default: `./results`.       |
