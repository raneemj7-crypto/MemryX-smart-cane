#!/bin/bash

echo "Installing GUI dependencies for VisionGuide..."
echo "=============================================="

# Install tkinter and PIL
sudo apt-get update
sudo apt-get install -y python3-tk python3-pil python3-pil.imagetk

# Verify installation
python3 -c "import tkinter; print('tkinter: OK')" 2>/dev/null && echo "[OK] tkinter installed" || echo "[FAIL] tkinter installation failed"
python3 -c "from PIL import Image, ImageTk; print('PIL: OK')" 2>/dev/null && echo "[OK] PIL installed" || echo "[FAIL] PIL installation failed"

echo ""
echo "Installation complete!"
echo "Now you can run: python3 visionguide_complete.py"
