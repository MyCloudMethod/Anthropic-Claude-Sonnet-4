#!/usr/bin/env python3
"""
Basic usage examples for the Claude-like language model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.transformer import ClaudeLikeTransformer, count_parameters
from utils.tokenizer import create_tokenizer
from utils.data_processing import create_sample_data, ConversationDataset, create_dataloader
from training.trainer import ModelTrainer
from inference.chat_interface import ClaudeLikeChat


def example_1_model_initialization():
    """Example 1: Initialize and test the model"""
    print("=" * 50)
    print("Example 1: Model Initialization")
    print("=" * 50)
    
    # Create tokenizer
    tokenizer = create_tokenizer("gpt2")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Initialize model
    model = ClaudeLikeTransformer(
        vocab_size=len(tokenizer),
        d_model=512,  # Smaller for demo
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=1024
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Output dtype: {logits.dtype}")


def example_2_tokenization():
    """Example 2: Tokenization examples"""
    print("\n" + "=" * 50)
    print("Example 2: Tokenization")
    print("=" * 50)
    
    tokenizer = create_tokenizer("gpt2")
    
    # Basic tokenization
    text = "Hello, I'm Claude, an AI assistant created by Anthropic."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    # Conversation format
    conversation = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data."}
    ]
    
    conv_tokens = tokenizer.encode_conversation(conversation)
    conv_decoded = tokenizer.decode_conversation(conv_tokens)
    
    print(f"\nOriginal conversation: {conversation}")
    print(f"Conversation tokens: {conv_tokens}")
    print(f"Decoded conversation: {conv_decoded}")


def example_3_data_processing():
    """Example 3: Data processing"""
    print("\n" + "=" * 50)
    print("Example 3: Data Processing")
    print("=" * 50)
    
    # Create sample data
    create_sample_data()
    print("Sample data created")
    
    # Load dataset
    tokenizer = create_tokenizer("gpt2")
    dataset = ConversationDataset("data/train.jsonl", tokenizer, max_length=512)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    
    # Decode to see content
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"Decoded sample: {decoded[:200]}...")
    
    # Create dataloader
    dataloader = create_dataloader(dataset, tokenizer, batch_size=2)
    batch = next(iter(dataloader))
    
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")


def example_4_generation():
    """Example 4: Text generation (requires trained model)"""
    print("\n" + "=" * 50)
    print("Example 4: Text Generation")
    print("=" * 50)
    
    # Initialize a small model for demo
    tokenizer = create_tokenizer("gpt2")
    model = ClaudeLikeTransformer(
        vocab_size=len(tokenizer),
        d_model=256,  # Very small for demo
        n_heads=4,
        n_layers=3,
        d_ff=1024,
        max_seq_len=512
    )
    
    # Generate with random weights (won't be coherent, just for demo)
    prompt = "Hello, I am"
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    print(f"Prompt: {prompt}")
    
    with torch.no_grad():
        # Generate a few tokens
        output = model.generate(
            input_ids,
            max_length=input_ids.size(1) + 10,
            temperature=1.0,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(output[0])
    print(f"Generated (untrained): {generated_text}")
    print("Note: This model is untrained, so output will be random!")


def example_5_training_setup():
    """Example 5: Training setup (without actually training)"""
    print("\n" + "=" * 50)
    print("Example 5: Training Setup")
    print("=" * 50)
    
    try:
        # Initialize trainer (will create sample data if needed)
        trainer = ModelTrainer("config.yaml")
        
        print(f"Trainer initialized")
        print(f"Device: {trainer.device}")
        print(f"Model parameters: {count_parameters(trainer.model):,}")
        print(f"Optimizer: {type(trainer.optimizer).__name__}")
        
        print("To actually train, run: python scripts/train.py")
        
    except Exception as e:
        print(f"Training setup error: {e}")
        print("Make sure config.yaml exists and is valid")


def example_6_chat_interface():
    """Example 6: Chat interface setup"""
    print("\n" + "=" * 50)
    print("Example 6: Chat Interface")
    print("=" * 50)
    
    print("Chat interface requires a trained model checkpoint.")
    print("After training, you can use:")
    print("  python scripts/chat.py outputs/best_model.pt --mode cli")
    print("  python scripts/chat.py outputs/best_model.pt --mode gradio")
    print("  python scripts/chat.py outputs/best_model.pt --mode api")
    
    print("\nFor API usage:")
    print("  curl -X POST http://localhost:7860/chat \\")
    print("    -H 'Content-Type: application/json' \\") 
    print("    -d '{\"message\": \"Hello\", \"temperature\": 0.7}'")


def main():
    """Run all examples"""
    print("Claude-like Language Model - Usage Examples")
    print("=" * 70)
    
    try:
        example_1_model_initialization()
        example_2_tokenization()
        example_3_data_processing()
        example_4_generation()
        example_5_training_setup()
        example_6_chat_interface()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("Next steps:")
        print("1. Train the model: python scripts/train.py")
        print("2. Chat with model: python scripts/chat.py outputs/best_model.pt")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()