import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union, Optional
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class ConversationDataset(Dataset):
    """Dataset for conversation-style training data"""
    
    def __init__(
        self, 
        data_file: str, 
        tokenizer, 
        max_length: int = 1024,
        include_system: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_system = include_system
        
        # Load and process data
        self.conversations = self.load_conversations(data_file)
        print(f"Loaded {len(self.conversations)} conversations from {data_file}")
    
    def load_conversations(self, data_file: str) -> List[List[Dict[str, str]]]:
        """Load conversations from various file formats"""
        conversations = []
        file_path = Path(data_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        if file_path.suffix == '.jsonl':
            # JSONL format: each line is a conversation
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading conversations"):
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        if 'conversation' in data:
                            conversations.append(data['conversation'])
                        elif 'messages' in data:
                            conversations.append(data['messages'])
                        else:
                            # Assume the entire line is a conversation
                            conversations.append(data)
        
        elif file_path.suffix == '.json':
            # JSON format: array of conversations
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    conversations = data
                else:
                    conversations = [data]
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return conversations
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Add system message if not present and include_system is True
        if self.include_system and (not conversation or conversation[0]["role"] != "system"):
            system_message = {
                "role": "system",
                "content": "You are a helpful, harmless, and honest AI assistant."
            }
            conversation = [system_message] + conversation
        
        # Encode conversation
        token_ids = self.tokenizer.encode_conversation(conversation)
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            # Pad with pad_token_id
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in token_ids]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(token_ids, dtype=torch.long)  # For language modeling, labels = input_ids
        }


class TextDataset(Dataset):
    """Dataset for plain text training data"""
    
    def __init__(
        self, 
        data_file: str, 
        tokenizer, 
        max_length: int = 1024,
        stride: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Load and process text
        self.text_chunks = self.load_and_chunk_text(data_file)
        print(f"Created {len(self.text_chunks)} text chunks from {data_file}")
    
    def load_and_chunk_text(self, data_file: str) -> List[List[int]]:
        """Load text file and create overlapping chunks"""
        file_path = Path(data_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Read entire text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize entire text
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Create overlapping chunks
        chunks = []
        for i in range(0, len(token_ids), self.stride):
            chunk = token_ids[i:i + self.max_length]
            if len(chunk) >= self.max_length // 2:  # Only keep chunks that are at least half full
                chunks.append(chunk)
        
        return chunks
    
    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        token_ids = self.text_chunks[idx]
        
        # Pad if necessary
        if len(token_ids) < self.max_length:
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
        
        # Create attention mask
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in token_ids]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(token_ids, dtype=torch.long)
        }


class DataCollator:
    """Data collator for batching"""
    
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, batch):
        # Get maximum sequence length in batch
        max_length = max(len(item['input_ids']) for item in batch)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad all sequences to max_length
        padded_batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for item in batch:
            input_ids = item['input_ids'].tolist()
            attention_mask = item['attention_mask'].tolist()
            labels = item['labels'].tolist()
            
            # Pad sequences
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                labels += [-100] * padding_length  # -100 is ignored in loss computation
            
            padded_batch['input_ids'].append(input_ids)
            padded_batch['attention_mask'].append(attention_mask)
            padded_batch['labels'].append(labels)
        
        # Convert to tensors
        return {
            'input_ids': torch.tensor(padded_batch['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(padded_batch['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(padded_batch['labels'], dtype=torch.long)
        }


def create_sample_data():
    """Create sample training data for testing"""
    sample_conversations = [
        {
            "conversation": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."}
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "How does a neural network work?"},
                {"role": "assistant", "content": "A neural network works by processing information through layers of interconnected nodes (neurons). Each connection has a weight, and the network learns by adjusting these weights based on training data."}
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "What are the benefits of renewable energy?"},
                {"role": "assistant", "content": "Renewable energy sources like solar and wind power offer several benefits: they reduce greenhouse gas emissions, provide sustainable energy, create jobs, and decrease dependence on fossil fuels."}
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "Explain quantum computing in simple terms."},
                {"role": "assistant", "content": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously, potentially solving certain problems much faster."}
            ]
        }
    ]
    
    # Create sample training data
    train_data_path = Path("data/train.jsonl")
    train_data_path.parent.mkdir(exist_ok=True)
    
    with open(train_data_path, 'w', encoding='utf-8') as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv) + '\n')
    
    # Create sample evaluation data (smaller subset)
    eval_data_path = Path("data/eval.jsonl")
    with open(eval_data_path, 'w', encoding='utf-8') as f:
        for conv in sample_conversations[:2]:  # Use first 2 conversations for eval
            f.write(json.dumps(conv) + '\n')
    
    print(f"Created sample data: {train_data_path} and {eval_data_path}")


def create_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_to_multiple_of: Optional[int] = None
) -> DataLoader:
    """Create a DataLoader with appropriate collation"""
    collator = DataCollator(tokenizer, pad_to_multiple_of)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    # Create sample data
    create_sample_data()
    
    # Test the dataset
    from utils.tokenizer import ClaudeTokenizer
    
    tokenizer = ClaudeTokenizer()
    
    # Test ConversationDataset
    dataset = ConversationDataset("data/train.jsonl", tokenizer, max_length=512)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")
    
    # Decode the sample to see what it looks like
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"Decoded sample: {decoded}")
    
    # Test DataLoader
    dataloader = create_dataloader(dataset, tokenizer, batch_size=2)
    batch = next(iter(dataloader))
    
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")