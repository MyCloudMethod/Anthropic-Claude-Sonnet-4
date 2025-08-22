import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import json
import time
import math
from pathlib import Path
from typing import Dict, Optional, Union
from tqdm import tqdm
import wandb
from transformers import get_cosine_schedule_with_warmup
import numpy as np

from model.transformer import ClaudeLikeTransformer, count_parameters
from utils.tokenizer import ClaudeTokenizer, create_tokenizer
from utils.data_processing import ConversationDataset, create_dataloader


class ModelTrainer:
    """Complete training pipeline for the Claude-like language model"""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = create_tokenizer("gpt2")
        print(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
        
        # Update config with actual vocab size
        self.config['model']['vocab_size'] = len(self.tokenizer)
        
        # Initialize model
        self.model = ClaudeLikeTransformer(**self.config['model'])
        self.model.to(self.device)
        
        print(f"Model initialized with {count_parameters(self.model):,} parameters")
        
        # Initialize optimizer
        self.optimizer = self.create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(self.config['logging']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize wandb if specified
        if 'wandb_project' in self.config['logging'] and self.config['logging']['wandb_project']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config,
                name=f"claude-like-{int(time.time())}"
            )
            wandb.watch(self.model)
    
    def create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay"""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norm parameters
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config['training']['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return optim.AdamW(
            optimizer_groups,
            lr=self.config['training']['learning_rate'],
            betas=(0.9, 0.95),
            eps=1e-8
        )
    
    def create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids, attention_mask)
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute cross-entropy loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(shift_logits, shift_labels)
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                loss = self.compute_loss(batch)
                batch_size = batch['input_ids'].size(0)
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss)
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved at step {self.step}")
        
        # Keep only the last few checkpoints to save space
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save disk space"""
        checkpoints = list(self.output_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > keep_last:
            # Sort by step number and keep only the latest ones
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from step {self.step}")
    
    def generate_sample(self, prompt: str, max_length: int = 100) -> str:
        """Generate a sample text for monitoring training progress"""
        self.model.eval()
        
        with torch.no_grad():
            # Encode prompt
            input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
            
            # Generate
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop"""
        # Load checkpoint if resuming
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Load datasets
        print("Loading datasets...")
        train_dataset = ConversationDataset(
            self.config['data']['train_file'],
            self.tokenizer,
            max_length=self.config['data']['max_length']
        )
        
        eval_dataset = ConversationDataset(
            self.config['data']['eval_file'],
            self.tokenizer,
            max_length=self.config['data']['max_length']
        )
        
        # Create dataloaders
        train_dataloader = create_dataloader(
            train_dataset,
            self.tokenizer,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        eval_dataloader = create_dataloader(
            eval_dataset,
            self.tokenizer,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        # Create scheduler
        num_training_steps = self.config['training']['max_steps']
        self.create_scheduler(num_training_steps)
        
        print(f"Starting training from step {self.step}")
        print(f"Training for {num_training_steps} steps")
        
        # Training loop
        while self.step < num_training_steps:
            epoch_loss = 0
            epoch_steps = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.epoch}")
            
            for batch in progress_bar:
                # Training step
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                epoch_steps += 1
                self.step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'ppl': f"{metrics['perplexity']:.2f}",
                    'lr': f"{metrics['learning_rate']:.2e}"
                })
                
                # Log metrics
                if wandb.run:
                    wandb.log({
                        'train_loss': metrics['loss'],
                        'train_perplexity': metrics['perplexity'],
                        'learning_rate': metrics['learning_rate'],
                        'step': self.step
                    })
                
                # Evaluation
                if self.step % self.config['training']['eval_steps'] == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    print(f"\nStep {self.step} - Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                          f"Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                    
                    # Log eval metrics
                    if wandb.run:
                        wandb.log(eval_metrics)
                    
                    # Save checkpoint if best model
                    is_best = eval_metrics['eval_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = eval_metrics['eval_loss']
                    
                    if self.step % self.config['training']['save_steps'] == 0:
                        self.save_checkpoint(is_best=is_best)
                
                # Generate sample text
                if self.step % (self.config['training']['eval_steps'] * 2) == 0:
                    sample_text = self.generate_sample("Hello, I am")
                    print(f"\nGenerated sample: {sample_text}\n")
                    
                    if wandb.run:
                        wandb.log({'generated_sample': sample_text})
                
                # Check if we've reached max steps
                if self.step >= num_training_steps:
                    break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"Epoch {self.epoch} completed. Average loss: {avg_epoch_loss:.4f}")
            
            self.epoch += 1
        
        # Save final checkpoint
        self.save_checkpoint(is_best=False)
        print("Training completed!")
        
        if wandb.run:
            wandb.finish()


def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Claude-like language model")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create sample data if it doesn't exist
    from utils.data_processing import create_sample_data
    if not Path("data/train.jsonl").exists():
        print("Creating sample training data...")
        create_sample_data()
    
    # Initialize trainer
    trainer = ModelTrainer(args.config)
    
    # Start training
    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()