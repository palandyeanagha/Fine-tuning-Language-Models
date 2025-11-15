#!/bin/bash

# 1. Create a pyenv virtual environment with a specific Python version
pyenv virtualenv 3.9 myenv   # choose Python 3.9 for compatibility with torch 1.13.1

# 2. Activate the virtual environment
pyenv activate myenv

# 3. Install all required libraries from requirements.txt
pip install -r requirements.txt

# 4. Download NLTK resources
python -m nltk.downloader wordnet punkt
