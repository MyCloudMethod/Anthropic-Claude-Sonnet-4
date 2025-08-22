# Quick Start Guide

Get up and running with your Claude-like language model in minutes!

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Examples (Optional)
```bash
python examples/basic_usage.py
```

### 3. Start Training
```bash
# Creates sample data automatically
python scripts/train.py --config config.yaml
```

### 4. Chat with Your Model
```bash
# After training completes
python scripts/chat.py outputs/best_model.pt
```

## ⚡ One-Liner Commands

### Training
```bash
# Basic training
python scripts/train.py

# Resume from checkpoint
python scripts/train.py --resume outputs/checkpoint_step_1000.pt
```

### Chat Interfaces
```bash
# Command line
python scripts/chat.py outputs/best_model.pt --mode cli

# Web interface
python scripts/chat.py outputs/best_model.pt --mode gradio --port 7860

# REST API
python scripts/chat.py outputs/best_model.pt --mode api --port 8000
```

## 📊 Configuration Quick Reference

Edit `config.yaml` to customize your model:

```yaml
# Model size (adjust for your hardware)
model:
  d_model: 768     # 768=small, 1024=medium, 1600=large
  n_layers: 12     # 6=small, 12=medium, 24=large
  
# Training
training:
  batch_size: 32   # Reduce if out of memory
  max_steps: 10000 # Increase for better quality
```

## 🛠️ Common Model Sizes

| Size | d_model | n_layers | Parameters | VRAM (Training) |
|------|---------|----------|------------|------------------|
| Tiny | 384     | 6        | ~30M       | ~1GB            |
| Small| 768     | 12       | ~124M      | ~2GB            |
| Medium| 1024   | 24       | ~350M      | ~4GB            |
| Large| 1600    | 48       | ~1.5B      | ~16GB           |

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` in config |
| Slow training | Use GPU, increase `batch_size` |
| Poor quality | Increase `max_steps`, better data |
| Import errors | Check Python path, install deps |

## 📁 Project Structure After Setup

```
claude-like-model/
├──  config.yaml          # ← Edit this for your needs
├── 📁 data/                # Training data (auto-created)
├── 📁 outputs/             # Model checkpoints
├── 📁 model/               # Model architecture
├── 📁 utils/               # Tokenizer & data processing
├── 📁 training/            # Training pipeline
├── 📁 inference/           # Chat interfaces
└── 📁 scripts/             # Entry points
    ├──  train.py         # ← Start here
    └──  chat.py          # ← Use after training
```

## 🎯 Next Steps

1. **Customize your data**: Replace files in `data/` with your own conversations
2. **Adjust model size**: Edit `config.yaml` based on your hardware
3. **Monitor training**: Check `outputs/` folder for checkpoints
4. **Deploy**: Use the API mode for production deployment

## 💡 Tips

- Start with a small model to test everything works
- Use Weights & Biases for training monitoring
- Save checkpoints frequently during long training runs
- Experiment with different temperature values during inference

Happy modeling! 🤖