#!/usr/bin/env python3
"""
Test script for the Product Research Assistant agent.
This will run all example questions to verify they work correctly.
"""

import os
import sys

# Set test mode environment variable
os.environ["TEST_MODE"] = "true"

# Import and run the main function from the chat module
sys.path.append("ChatModels")
from chat import main

if __name__ == "__main__":
    print("Running Product Research Assistant tests...")
    main()
    print("Tests completed.") 