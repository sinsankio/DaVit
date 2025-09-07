# DaVit - Vision Transformer-Powered Self-Driving Car

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue)](https://python-poetry.org/)

DaVit is an innovative implementation of Vision Transformers (ViT) for autonomous driving, specifically designed to work with the Udacity self-driving car simulator. This project demonstrates the power of attention mechanisms in computer vision tasks by applying them to **behavioral cloning** for autonomous vehicle control.

## Real Time Simulation (Autonomus mode)

<p align="center">
  <img src="assets/vit_live.gif" alt="DaVit in Action" width="100%">
</p>

## What is DaVit?

DaVit combines the power of Vision Transformers with behavioral cloning to create a robust self-driving system. The project uses a pre-trained ViT model (`google/vit-base-patch16-224`) and fine-tunes it for steering angle prediction based on camera input from the simulator.

## Development Architecture

DaVit is developed using a two-phase approach:

### 1. Development Phase (`dev` directory)
- **Data preprocessing utilities** - Robust data handling  tools
- **Model training infrastructure** - Complete training pipeline setup
- **Jupyter notebooks** - Interactive data/model experimentation
- **Model checkpoints** - Archive for both training and validation models

### 2. Application Phase (`app` directory)
- **Pre-trained model checkpoints** - Ready-to-use trained models
- **Production-ready implementation** - Optimized for real-time performance
- **Drive server implementation** - Real-time inference engine
- **Udacity self-driving simulator** - Bundled simulator environment

## Key Features

- **Vision Transformer Integration**: Utilizes `google/vit-base-patch16-224` for advanced image reasoning
- **Data Augmentation for Wider Context Awareness**: 
  - Image resizing and cropping
  - Random brightness adjustment
  - Shadow augmentation
  - Shear transformation
- **Real-time Inference**: Processes simulator camera feed for live steering predictions
- **Telemetry Recording**: Captures comprehensive driving data during realtime execution and exports to video (.mp4) format for post-processing
- **Adaptive Speed Control**: Implements dynamic speed adjustment based on steering angles

## Installation

### Prerequisites

- ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
- ![Poetry](https://img.shields.io/badge/Poetry-dependency%20management-60A5FA?style=flat&logo=poetry&logoColor=white)
- 🎮 CUDA-capable GPU (recommended for training)

```bash
# Clone the repository
git clone https://github.com/sinsankio/DaVit.git
cd DaVit

# Download fine-tuned ViT model for app inference
# View file app/chkpts/vit_v1_drive.txt for download instructions
 
# Set up Python environment using Poetry
poetry install
poetry sync

# Activate the environment (optional: explicit env activation)
$(poetry env activate)

# The simulator is already included in app/simulator-windows-64/

# Execute driver application via Python (if poetry env is explicitly activated) 
python app\scripts\drive.py

or else

# Execute driver application directly via Poetry
poetry run app\scripts\drive.py
```

## Usage

### Drive Server Configuration

The drive server (`app/scripts/drive.py`) supports various command-line arguments:

```bash
python app/scripts/drive.py [OPTIONS]
```

#### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max_speed` | INT | - | Maximum speed limit |
| `--min_speed` | INT | - | Minimum speed limit |
| `--host` | STR | `localhost` | Host address |
| `--port` | INT | `4567` | Port number |
| `--validation` | FLAG | - | Enable validation mode for testing |

#### Example Usage

```bash
# Basic usage with default settings
python app/scripts/drive.py

# Custom speed limits and port
python app/scripts/drive.py --max_speed 25 --min_speed 10 --port 8080

# Validation mode
python app/scripts/drive.py --validation
```

## Future Improvements

- [ ] **Multi-camera Support** - Integration with stereo camera setups
- [ ] **Environment Expansion** - Support for additional simulator environments
- [ ] **Enhanced Augmentation** - Advanced data augmentation techniques
- [ ] **Real-world Testing** - Bridge to real-world autonomous driving scenarios
- [ ] **Object Detection** - Integration of object detection and tracking capabilities
- [ ] **Model Optimization** - Performance improvements and model compression

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Create** a Pull Request

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

</div>
