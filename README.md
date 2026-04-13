# ViT-SOT: One-Stream Vision Transformer for Single Object Tracking

A clean, well-documented PyTorch implementation of **OSTrack** (One-Stream Tracking with ViT) trained on the **GOT-10k** benchmark.

> **Paper**: [One-Stream Exploration-Exploitation Network for Single Object Tracking](https://arxiv.org/abs/2203.11991)  
> **Ye et al., ECCV 2022**

---

## What is OSTrack?

OSTrack is a state-of-the-art single object tracker that:

1. **Processes template + search jointly** through a Vision Transformer — unlike two-stream trackers (SiamRPN, TransT), there is no separate fusion module. The standard self-attention mechanism naturally enables cross-attention between the template and search tokens from the very first layer.

2. **Uses a pretrained ViT backbone** (ViT-Small or ViT-Base from ImageNet) — the rich features learned on ImageNet transfer extremely well to tracking.

3. **Predicts bounding boxes** with a simple convolutional head operating on the 2D reshaped search-region tokens.

```
Template (128×128) → patch embed (64 tokens)  ─┐
                                                 concat [320 tokens] → ViT (12 layers) → Head → bbox
Search   (256×256) → patch embed (256 tokens) ─┘
```

## Performance (GOT-10k test set)

| Model | AO | SR₀.₅ | SR₀.₇₅ | Speed |
|-------|-----|-------|--------|-------|
| SiamFC | 0.374 | 0.404 | 0.144 | 86 FPS |
| SiamRPN++ | 0.518 | 0.616 | 0.325 | 35 FPS |
| TransT | 0.671 | 0.762 | 0.576 | 50 FPS |
| **OSTrack-256** | **0.711** | **0.804** | **0.617** | **105 FPS** |
| **OSTrack-384** | **0.739** | **0.836** | **0.654** | 58 FPS |

## Project Structure

```
vit-sot-got10k/
├── configs/
│   ├── ostrack_small.yaml      # ViT-Small (fast, good for experiments)
│   └── ostrack_base.yaml       # ViT-Base  (best performance)
├── lib/
│   ├── models/
│   │   ├── vit_backbone.py     # ViT backbone (one-stream, patch embed, pos embed)
│   │   ├── head.py             # CornerHead + losses (L1, GIoU, Focal)
│   │   └── ostrack.py          # Full OSTrack model + factory functions
│   ├── datasets/
│   │   ├── got10k.py           # GOT-10k dataset, sequence reader, crop strategy
│   │   └── transforms.py       # Color jitter, flip augmentation
│   ├── tracking/
│   │   └── tracker.py          # Stateful online tracker (init + track)
│   └── utils/
│       ├── box_ops.py          # IoU, coordinate conversions
│       ├── metrics.py          # AO, SR50, SR75, AUC, Precision
│       └── visualize.py        # Tracking visualization tools
├── notebooks/
│   └── 01_vit_sot_tutorial.ipynb  # Full interactive tutorial
├── scripts/
│   └── download_got10k.py      # Dataset download instructions
├── train.py                    # Training script (AMP, cosine LR, grad clip)
├── evaluate.py                 # Evaluation (per-sequence metrics + summary)
└── demo.py                     # Demo on your own video
```

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision timm pyyaml matplotlib pillow tqdm opencv-python
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Download GOT-10k Dataset

```bash
# Get download instructions
python scripts/download_got10k.py --info

# After manual download, verify structure
python scripts/download_got10k.py --verify --output data/got10k
```

The dataset should be organized as:
```
data/got10k/
├── train/   (10,000 sequences, ~35GB)
├── val/     (180 sequences, ~400MB)
└── test/    (180 sequences, ~1.5GB)
```

## Training

```bash
# Train with ViT-Small (recommended for experiments, ~12GB GPU RAM)
python train.py --config configs/ostrack_small.yaml

# Train with ViT-Base (best performance, ~40GB GPU RAM or reduce batch_size)
python train.py --config configs/ostrack_base.yaml

# Resume from checkpoint
python train.py --config configs/ostrack_small.yaml --resume checkpoints/epoch_050.pth

# Debug mode (1 epoch, 10 batches — verifies code runs)
python train.py --config configs/ostrack_small.yaml --debug

# Multi-GPU
python train.py --config configs/ostrack_base.yaml --gpus 0,1,2,3
```

Training details:
- **Optimizer**: AdamW with layer-wise LR (`backbone_lr = base_lr × 0.1`)
- **Schedule**: Cosine decay with 2-epoch linear warmup
- **Augmentation**: Color jitter (independent for template/search), horizontal flip
- **Mixed Precision**: Enabled by default (`use_amp: true`)
- **Epochs**: 300 (standard GOT-10k protocol)

## Evaluation

```bash
# Evaluate on GOT-10k val set
python evaluate.py --checkpoint output/ostrack_small/best.pth --split val

# Evaluate on test set (for submission to GOT-10k server)
python evaluate.py --checkpoint output/ostrack_small/best.pth --split test --save_results

# Single sequence with visualization
python evaluate.py --checkpoint output/ostrack_small/best.pth \
                   --seq GOT-10k_Val_000001 --visualize
```

## Demo

```bash
# Track an object in your own video (draws bbox in first frame)
python demo.py --checkpoint output/ostrack_small/best.pth --video my_video.mp4

# With a known initial bbox
python demo.py --checkpoint output/ostrack_small/best.pth \
               --video video.mp4 --init_box 100,150,300,400

# Webcam
python demo.py --checkpoint output/ostrack_small/best.pth --webcam
```

## Interactive Tutorial

Open the Jupyter notebook for a step-by-step walkthrough:

```bash
jupyter notebook notebooks/01_vit_sot_tutorial.ipynb
```

Topics covered:
- SOT problem definition and challenges
- Evolution: SiamFC → SiamRPN → OSTrack
- ViT anatomy: patch embedding, positional encoding, attention
- GOT-10k dataset and crop strategy
- Model forward pass (step by step with tensor shapes)
- Loss functions (L1, GIoU, Focal) — theory and code
- Training strategy (LR decay, AMP, gradient clipping)
- Evaluation metrics (AO, SR50, SR75, Success curve)
- Attention map visualization

## Key Design Choices Explained

### Why one-stream?

Traditional two-stream trackers process template and search through separate (though shared-weight) networks, then fuse features late. OSTrack processes both through the **same transformer layers simultaneously**. This enables:

- **Earlier interaction**: Every ViT layer can correlate template and search features
- **Simpler architecture**: No explicit cross-attention module needed
- **Better performance**: More interaction layers → richer features

### Why ViT instead of CNN?

- ViT's global receptive field (from the first layer) is better for tracking than CNN's local receptive field
- ImageNet-pretrained ViT features generalize well to unseen object classes
- No need for position-specific normalization (BN is tricky with tracking crops of varying sizes)

### Crop strategy

Following SiamFC, we crop a square region around the object:
```
crop_size = sqrt((w + context) * (h + context))
where context = (w + h) / 2 * factor
```
- Template: `factor=2.0` — tight crop, mostly the object
- Search: `factor=4.0` — wide crop, includes surroundings for motion

This ensures the object is **always centered** in the crop, decoupling tracking from absolute image position.

## Citation

If you use this code, please cite the original OSTrack paper:

```bibtex
@inproceedings{ye2022ostrack,
  title={Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework},
  author={Ye, Botao and Chang, Hong and Ma, Bingpeng and Shan, Shiguang and Chen, Xilin},
  booktitle={ECCV},
  year={2022}
}
```

And the GOT-10k benchmark:
```bibtex
@article{huang2019got,
  title={Got-10k: A large high-diversity benchmark for generic object tracking in the wild},
  author={Huang, Lianghua and Zhao, Xin and Huang, Kaiqi},
  journal={TPAMI},
  year={2021}
}
```
