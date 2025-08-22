import torch
import yaml
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import gradio as gr
from flask import Flask, request, jsonify
import threading

from model.transformer import ClaudeLikeTransformer
from utils.tokenizer import create_tokenizer


class ClaudeLikeChat:
    """Chat interface for the trained Claude-like model"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # Override config if provided
        if config_path:
            with open(config_path, 'r') as f:
                config_override = yaml.safe_load(f)
                self.config.update(config_override)
        
        # Initialize tokenizer
        self.tokenizer = create_tokenizer("gpt2")
        
        # Initialize model
        self.model = ClaudeLikeTransformer(**self.config['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Conversation history
        self.conversation_history = []
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def add_system_message(self, content: str):
        """Add a system message to the conversation"""
        self.conversation_history.insert(0, {"role": "system", "content": content})
    
    def generate_response(
        self,
        user_message: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        """Generate a response to the user message"""
        
        # Add user message to conversation
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Add assistant role marker (the model will complete this)
        conversation_with_assistant = self.conversation_history + [{"role": "assistant", "content": ""}]
        
        # Encode conversation
        input_ids = self.tokenizer.encode_conversation(conversation_with_assistant[:-1])
        input_ids.append(self.tokenizer.assistant_token_id)  # Add assistant token to prompt completion
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_length=min(len(input_ids) + max_length, self.config['model']['max_seq_len']),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode the generated part only
        generated_ids = output[0][len(input_ids):].tolist()
        
        # Stop at end of text or user token
        stop_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.user_token_id,
            self.tokenizer.system_token_id
        ]
        
        # Find first stop token
        for i, token_id in enumerate(generated_ids):
            if token_id in stop_tokens:
                generated_ids = generated_ids[:i]
                break
        
        # Decode response
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Apply simple repetition penalty (remove repeated phrases)
        response = self._apply_repetition_penalty(response, repetition_penalty)
        
        # Add assistant response to conversation
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _apply_repetition_penalty(self, text: str, penalty: float) -> str:
        """Simple repetition penalty to reduce repeated phrases"""
        if penalty <= 1.0:
            return text
        
        words = text.split()
        if len(words) < 4:
            return text
        
        # Look for repeated 2-3 word phrases
        cleaned_words = []
        i = 0
        
        while i < len(words):
            current_word = words[i]
            
            # Check for immediate repetition
            if i > 0 and current_word == words[i-1]:
                i += 1
                continue
            
            # Check for phrase repetition
            if i < len(words) - 2:
                phrase = " ".join(words[i:i+2])
                next_phrase_start = i + 2
                
                # Look for the same phrase appearing soon after
                found_repetition = False
                for j in range(next_phrase_start, min(next_phrase_start + 6, len(words) - 1)):
                    if j < len(words) - 1:
                        next_phrase = " ".join(words[j:j+2])
                        if phrase == next_phrase:
                            found_repetition = True
                            break
                
                if found_repetition:
                    i += 2  # Skip the repeated phrase
                    continue
            
            cleaned_words.append(current_word)
            i += 1
        
        return " ".join(cleaned_words)
    
    def chat_loop(self):
        """Command line chat loop"""
        print("Claude-like Chat Interface")
        print("Type 'quit' to exit, 'reset' to reset conversation, 'system <message>' to add system message")
        print("-" * 50)
        
        # Add default system message
        self.add_system_message("You are a helpful, harmless, and honest AI assistant.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'reset':
                    self.reset_conversation()
                    self.add_system_message("You are a helpful, harmless, and honest AI assistant.")
                    print("Conversation reset.")
                    continue
                elif user_input.lower().startswith('system '):
                    system_message = user_input[7:]  # Remove 'system ' prefix
                    self.reset_conversation()
                    self.add_system_message(system_message)
                    print(f"System message set: {system_message}")
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                print("\nAssistant: ", end="", flush=True)
                response = self.generate_response(
                    user_input,
                    max_length=300,
                    temperature=0.7,
                    top_p=0.9
                )
                print(response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue


def create_gradio_interface(chat_model: ClaudeLikeChat):
    """Create Gradio web interface"""
    
    def chat_fn(message, history, temperature, max_length, top_p):
        # Convert gradio history format to our format
        chat_model.reset_conversation()
        chat_model.add_system_message("You are a helpful, harmless, and honest AI assistant.")
        
        for user_msg, assistant_msg in history:
            chat_model.conversation_history.append({"role": "user", "content": user_msg})
            chat_model.conversation_history.append({"role": "assistant", "content": assistant_msg})
        
        # Generate response
        response = chat_model.generate_response(
            message,
            max_length=int(max_length),
            temperature=temperature,
            top_p=top_p
        )
        
        return response
    
    def reset_fn():
        chat_model.reset_conversation()
        return []
    
    # Create interface
    with gr.Blocks(title="Claude-like Chat") as interface:
        gr.Markdown("# Claude-like Chat Interface")
        gr.Markdown("Chat with your trained language model!")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(label="Your message", placeholder="Type your message here...")
                
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                temperature = gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
                max_length = gr.Slider(50, 500, value=200, label="Max Length")
                top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top P")
        
        # Event handlers
        def user_message(user_message, history):
            return "", history + [[user_message, None]]
        
        def bot_response(history, temperature, max_length, top_p):
            user_message = history[-1][0]
            
            # Get bot response
            bot_message = chat_fn(user_message, history[:-1], temperature, max_length, top_p)
            history[-1][1] = bot_message
            
            return history
        
        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, temperature, max_length, top_p], chatbot
        )
        
        submit.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, temperature, max_length, top_p], chatbot
        )
        
        clear.click(reset_fn, outputs=[chatbot])
    
    return interface


def create_flask_api(chat_model: ClaudeLikeChat):
    """Create Flask API for the chat model"""
    
    app = Flask(__name__)
    
    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            data = request.json
            message = data.get('message', '')
            temperature = data.get('temperature', 0.7)
            max_length = data.get('max_length', 200)
            top_p = data.get('top_p', 0.9)
            reset = data.get('reset', False)
            
            if reset:
                chat_model.reset_conversation()
                chat_model.add_system_message("You are a helpful, harmless, and honest AI assistant.")
            
            if not message:
                return jsonify({'error': 'No message provided'}), 400
            
            response = chat_model.generate_response(
                message,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            return jsonify({
                'response': response,
                'conversation_history': chat_model.conversation_history
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/reset', methods=['POST'])
    def reset():
        try:
            chat_model.reset_conversation()
            chat_model.add_system_message("You are a helpful, harmless, and honest AI assistant.")
            return jsonify({'status': 'success', 'message': 'Conversation reset'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'model_loaded': True})
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Claude-like Chat Interface")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--config", help="Path to config file (optional)")
    parser.add_argument("--mode", choices=["cli", "gradio", "api"], default="cli",
                      help="Interface mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host for web interface")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    
    args = parser.parse_args()
    
    # Load chat model
    print("Loading model...")
    chat_model = ClaudeLikeChat(args.checkpoint, args.config)
    
    if args.mode == "cli":
        # Command line interface
        chat_model.chat_loop()
    
    elif args.mode == "gradio":
        # Gradio web interface
        interface = create_gradio_interface(chat_model)
        interface.launch(server_name=args.host, server_port=args.port, share=False)
    
    elif args.mode == "api":
        # Flask API
        app = create_flask_api(chat_model)
        print(f"Starting Flask API on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()