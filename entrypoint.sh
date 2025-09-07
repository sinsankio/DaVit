#!/bin/bash
set -e

# Start simulator in background
/app/simulator-windows-64/Default Windows desktop 64-bit.exe &

# Activate poetry venv and run driver
$(poetry env activate)
python app/scripts/drive.py
