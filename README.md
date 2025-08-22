# Claude-like Language Model

A complete implementation of a transformer-based language model similar to Claude, built from scratch using PyTorch. This project includes training pipeline, tokenization, data processing, and multiple inference interfaces.

## Features

- **Complete Transformer Architecture**: Multi-head attention, positional encoding, layer normalization
- **Flexible Tokenization**: Support for GPT-2 style BPE and tiktoken
- **Conversation Format**: Structured chat format with system, user, and assistant roles
- **Training Pipeline**: Full training loop with checkpointing, evaluation, and monitoring
- **Multiple Interfaces**: Command-line, Gradio web interface, and REST API
- **Generation Options**: Temperature, top-k, top-p sampling with repetition penalty

## Project Structure

```
claude-like-model/
├── config.yaml                 # Model and training configuration
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── model/
│   ├── __init__.py
│   └── transformer.py         # Transformer model implementation
├── utils/
│   ├── __init__.py
│   ├── tokenizer.py           # Tokenization utilities
│   └── data_processing.py     # Dataset and data loading
├── training/
│   ├── __init__.py
│   └── trainer.py             # Training pipeline
├── inference/
│   ├── __init__.py
│   └── chat_interface.py      # Chat interfaces
├── scripts/
│   ├── train.py              # Training script
│   └── chat.py               # Chat script
├── data/                     # Training data (created automatically)
└── outputs/                  # Model checkpoints and logs
```

## Quick Start

### 1. Installation

```bash
# Clone this repository
git clone <your-repo-url>
cd claude-like-model

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Train the model (creates sample data automatically)
python scripts/train.py --config config.yaml

# Resume training from checkpoint
python scripts/train.py --config config.yaml --resume outputs/checkpoint_step_1000.pt
```

### 3. Chat Interface

```bash
# Command line chat
python scripts/chat.py outputs/best_model.pt

# Web interface with Gradio
python scripts/chat.py outputs/best_model.pt --mode gradio

# REST API
python scripts/chat.py outputs/best_model.pt --mode api
```

## Configuration

The model behavior can be customized through `config.yaml`:

```yaml
model:
  vocab_size: 50257        # Vocabulary size
  d_model: 768            # Model dimension
  n_heads: 12             # Number of attention heads
  n_layers: 12            # Number of transformer layers
  d_ff: 3072              # Feed-forward dimension
  max_seq_len: 2048       # Maximum sequence length
  dropout: 0.1            # Dropout rate

training:
  batch_size: 32          # Training batch size
  learning_rate: 5e-4     # Learning rate
  max_steps: 100000       # Maximum training steps
  warmup_steps: 4000      # Warmup steps for learning rate
  save_steps: 1000        # Save checkpoint every N steps
  eval_steps: 500         # Evaluate every N steps
```

## Usage Examples

### Training Custom Data

1. Prepare your data in JSONL format:

```json
{"conversation": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"conversation": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thanks!"}]}
```

2. Update the data paths in `config.yaml`:

```yaml
data:
  train_file: "path/to/your/train.jsonl"
  eval_file: "path/to/your/eval.jsonl"
```

3. Start training:

```bash
python scripts/train.py --config config.yaml
```

### Using the Python API

```python
from inference.chat_interface import ClaudeLikeChat

# Load trained model
chat = ClaudeLikeChat("outputs/best_model.pt")

# Generate response
response = chat.generate_response(
    "What is artificial intelligence?",
    max_length=200,
    temperature=0.7
)
print(response)

# Reset conversation
chat.reset_conversation()
```

### REST API Usage

```bash
# Start API server
python scripts/chat.py outputs/best_model.pt --mode api --port 8000

# Send request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "temperature": 0.7, "max_length": 100}'
```

## Model Architecture

The model implements a standard transformer decoder architecture:

- **Multi-Head Attention**: Self-attention mechanism with multiple attention heads
- **Position-wise Feed-Forward**: Two-layer MLP with GELU activation
- **Layer Normalization**: Applied before each sub-layer (pre-norm)
- **Positional Encoding**: Sinusoidal position embeddings
- **Causal Masking**: Ensures autoregressive generation

### Key Features:

- **Conversation-aware**: Special tokens for system, user, and assistant roles
- **Efficient Generation**: Implements top-k, top-p sampling and temperature control
- **Scalable**: Configurable model size from small to large
- **Training Optimizations**: Gradient clipping, weight decay, cosine scheduling

## Performance Tips

### Training:
- Use mixed precision training for faster training: Add `torch.cuda.amp` integration
- Increase batch size and use gradient accumulation for better convergence
- Use multiple GPUs with `torch.nn.DataParallel` or `DistributedDataParallel`

### Inference:
- Use smaller models for faster inference
- Implement key-value caching for sequential generation
- Consider quantization for deployment

## Monitoring

The training pipeline includes comprehensive logging:

- **Weights & Biases**: Automatic experiment tracking (configure in `config.yaml`)
- **Metrics**: Loss, perplexity, learning rate, generated samples
- **Checkpoints**: Automatic saving of best models and regular checkpoints

## Advanced Usage

### Custom Tokenizer

```python
from utils.tokenizer import create_tokenizer

# Use tiktoken instead of GPT-2 tokenizer
tokenizer = create_tokenizer("tiktoken", encoding_name="cl100k_base")
```

### Model Sizes

Adjust model size in `config.yaml`:

```yaml
# Small model (~124M parameters)
model:
  d_model: 768
  n_heads: 12
  n_layers: 12
  d_ff: 3072

# Large model (~1.5B parameters)
model:
  d_model: 1600
  n_heads: 25
  n_layers: 48
  d_ff: 6400
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce `batch_size` or `max_seq_len` in config
2. **Slow Training**: Enable mixed precision, increase batch size, use multiple GPUs
3. **Poor Generation**: Adjust temperature, top-p values, or train for longer
4. **Repetitive Output**: Increase repetition penalty or use nucleus sampling

### Memory Requirements:

- Small model (124M params): ~2GB VRAM for training, ~500MB for inference
- Medium model (350M params): ~4GB VRAM for training, ~1GB for inference
- Large model (1.5B params): ~16GB VRAM for training, ~4GB for inference

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## Acknowledgments

This implementation is inspired by:
- The original Transformer paper "Attention Is All You Need"
- GPT and Claude architectures
- The Hugging Face Transformers library
- PyTorch documentation and examples

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{claude-like-model,
  title={Claude-like Language Model: A Complete Implementation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-username/claude-like-model}}
}