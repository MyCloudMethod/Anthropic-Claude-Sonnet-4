import json
import regex as re
from typing import List, Dict, Union
import tiktoken
from transformers import GPT2TokenizerFast
import torch


class ClaudeTokenizer:
    """A tokenizer similar to what Claude might use, based on BPE (Byte Pair Encoding)"""
    
    def __init__(self, vocab_file: str = None, merges_file: str = None, use_pretrained: bool = True):
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        
        self.special_tokens = [
            self.pad_token,
            self.unk_token, 
            self.bos_token,
            self.eos_token,
            self.system_token,
            self.user_token,
            self.assistant_token
        ]
        
        if use_pretrained and vocab_file is None:
            # Use GPT-2 tokenizer as base (similar to what many models use)
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            
            # Add special tokens
            special_tokens_dict = {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token,
            }
            
            additional_special_tokens = [
                self.system_token,
                self.user_token, 
                self.assistant_token
            ]
            
            self.tokenizer.add_special_tokens({
                **special_tokens_dict,
                'additional_special_tokens': additional_special_tokens
            })
            
        elif vocab_file and merges_file:
            # Load custom tokenizer
            self.tokenizer = GPT2TokenizerFast(
                vocab_file=vocab_file,
                merges_file=merges_file,
                **{token_type: token for token_type, token in special_tokens_dict.items()}
            )
        else:
            raise ValueError("Either use_pretrained=True or provide vocab_file and merges_file")
            
        self.vocab_size = len(self.tokenizer)
        
        # Token IDs for special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.system_token_id = self.tokenizer.convert_tokens_to_ids(self.system_token)
        self.user_token_id = self.tokenizer.convert_tokens_to_ids(self.user_token)
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids(self.assistant_token)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def encode_conversation(self, conversation: List[Dict[str, str]]) -> List[int]:
        """
        Encode a conversation in Claude-like format
        conversation: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
        """
        tokens = [self.bos_token_id]
        
        for message in conversation:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                role_token_id = self.system_token_id
            elif role == "user":
                role_token_id = self.user_token_id
            elif role == "assistant":
                role_token_id = self.assistant_token_id
            else:
                raise ValueError(f"Unknown role: {role}")
            
            tokens.append(role_token_id)
            tokens.extend(self.encode(content, add_special_tokens=False))
        
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode_conversation(self, token_ids: Union[List[int], torch.Tensor]) -> List[Dict[str, str]]:
        """Decode token IDs back to conversation format"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        conversation = []
        current_role = None
        current_tokens = []
        
        for token_id in token_ids:
            if token_id == self.system_token_id:
                if current_role and current_tokens:
                    conversation.append({
                        "role": current_role,
                        "content": self.decode(current_tokens, skip_special_tokens=True).strip()
                    })
                current_role = "system"
                current_tokens = []
            elif token_id == self.user_token_id:
                if current_role and current_tokens:
                    conversation.append({
                        "role": current_role,
                        "content": self.decode(current_tokens, skip_special_tokens=True).strip()
                    })
                current_role = "user"
                current_tokens = []
            elif token_id == self.assistant_token_id:
                if current_role and current_tokens:
                    conversation.append({
                        "role": current_role,
                        "content": self.decode(current_tokens, skip_special_tokens=True).strip()
                    })
                current_role = "assistant"
                current_tokens = []
            elif token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            else:
                current_tokens.append(token_id)
        
        # Add the last message
        if current_role and current_tokens:
            conversation.append({
                "role": current_role,
                "content": self.decode(current_tokens, skip_special_tokens=True).strip()
            })
        
        return conversation
    
    def __len__(self):
        return self.vocab_size


class TikTokenTokenizer:
    """Alternative tokenizer using tiktoken (similar to GPT-4)"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
        
        # Special tokens (tiktoken doesn't have built-in special tokens like transformers)
        self.pad_token = "<|pad|>"
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        
        # We'll use high token IDs for special tokens to avoid conflicts
        self.pad_token_id = self.vocab_size
        self.bos_token_id = self.vocab_size + 1
        self.eos_token_id = self.vocab_size + 2
        self.system_token_id = self.vocab_size + 3
        self.user_token_id = self.vocab_size + 4
        self.assistant_token_id = self.vocab_size + 5
        
        # Update vocab size to include special tokens
        self.vocab_size += 6
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.encoding.encode(text)
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if skip_special_tokens:
            # Filter out special tokens
            special_token_ids = {
                self.pad_token_id, self.bos_token_id, self.eos_token_id,
                self.system_token_id, self.user_token_id, self.assistant_token_id
            }
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]
        
        # Only decode regular tokens (not our custom special tokens)
        regular_tokens = [tid for tid in token_ids if tid < self.encoding.n_vocab]
        
        try:
            return self.encoding.decode(regular_tokens)
        except:
            return ""
    
    def __len__(self):
        return self.vocab_size


def create_tokenizer(tokenizer_type: str = "gpt2", **kwargs) -> Union[ClaudeTokenizer, TikTokenTokenizer]:
    """Factory function to create tokenizer"""
    if tokenizer_type == "gpt2":
        return ClaudeTokenizer(**kwargs)
    elif tokenizer_type == "tiktoken":
        return TikTokenTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = ClaudeTokenizer()
    
    # Test basic encoding/decoding
    text = "Hello, I'm Claude, an AI assistant created by Anthropic."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {len(tokenizer)}")
    
    # Test conversation encoding
    conversation = [
        {"role": "user", "content": "What is artificial intelligence?"},
        {"role": "assistant", "content": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior."}
    ]
    
    conv_encoded = tokenizer.encode_conversation(conversation)
    conv_decoded = tokenizer.decode_conversation(conv_encoded)
    
    print(f"\nOriginal conversation: {conversation}")
    print(f"Encoded conversation: {conv_encoded}")
    print(f"Decoded conversation: {conv_decoded}")