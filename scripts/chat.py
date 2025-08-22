#!/usr/bin/env python3
"""
Chat interface script for the Claude-like language model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.chat_interface import main

if __name__ == "__main__":
    main()