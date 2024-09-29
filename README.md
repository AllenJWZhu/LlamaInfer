# LLM Inference Engine

![ternsorrt](./imgs/inference-visual-tensor-rt-llm.png)

(The support for LlaMA 3.2 is still in progress)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Model Support](#model-support)
- [Advanced Features](#advanced-features)
- [License](#license)

## Overview

LLM Inference Engine is a high-performance, CUDA-accelerated framework for large language model inference. It's designed to efficiently run models like LLaMA and its variants on consumer-grade hardware, focusing on optimized memory usage and rapid text generation.

## Features

- CUDA-accelerated inference for optimal performance on NVIDIA GPUs
- Support for LLaMA and LLaMA2 model architectures
- Int8 quantization for reduced memory footprint
- Efficient KV-cache implementation for faster sequential inference
- Custom CUDA kernels for critical operations (RMSNorm, Softmax, SwiGLU, etc.)
- Memory-mapped weight loading for quick startup and reduced RAM usage
- Modular architecture allowing easy extension to new model types

## Architecture

The engine is built on a modular architecture with the following key components:

1. **Resource Manager**: Handles device memory allocation and tracking
2. **Tensor**: A custom implementation for n-dimensional arrays with GPU support
3. **Operator Registry**: Manages and dispatches computational operators
4. **Model Loader**: Efficiently loads model weights and parameters
5. **Inference Pipeline**: Orchestrates the forward pass through the model
6. **Quantization Module**: Implements int8 quantization for weights and activations

## Performance

On an NVIDIA RTX 3060 Laptop GPU, the engine achieves:

- 60.34 tokens/second for LLaMA 1.1B (FP32)

![](./imgs/do.gif)

## Dependencies

- CUDA Toolkit 11.4+
- cuBLAS
- Google glog
- Google Test (for unit tests)
- SentencePiece
- Armadillo (with OpenBLAS backend)

## Installation

Before starting the installation, ensure that your system has the following tools installed:

- CMake (version 3.10 or later)
- Docker (if using Docker-based builds)
- Git
- Build essentials (`make`, `gcc`, etc.)

### Step 1: Installing Mathematical Libraries

#### Armadillo Installation

[Armadillo](https://arma.sourceforge.net/) is a high-quality linear algebra library (C++). It provides a simple API while relying on [OpenBLAS](https://www.openblas.net/) for computational heavy-lifting.

Before installing Armadillo, the following dependencies must be installed:

```bash
sudo apt update
sudo apt install libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
Once the dependencies are installed, download the Armadillo source from one of the following locations:

## Usage

Run inference on a LLaMA model:

```bash
./llm_infer --model path/to/llama_7b.bin --tokenizer path/to/tokenizer.model --prompt "Once upon a time"
```

Once the dependencies are installed, download the Armadillo source from one of the following locations:

- Official download: https://arma.sourceforge.net/download.html
- Git mirror: https://gitee.com/mirrors/armadillo-code

After downloading the library:

```bash
# If cloning from the git repository
git clone https://gitee.com/mirrors/armadillo-code
cd armadillo-code

# Create build directory
mkdir build
cd build

# Build and install the library
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8  # Adjust '8' based on your CPU cores
sudo make install
```

For more details, visit [Armadillo Documentation](https://arma.sourceforge.net/docs.html).

### Step 2: Installing Unit Testing Library
Unit tests are crucial for verifying the correctness of the deep learning framework. For this project, we use the [Google Test](https://github.com/google/googletest) framework.

To install Google Test:

```bash
# Clone the Google Test repository
git clone https://github.com/google/googletest.git
cd googletest

# Create build directory
mkdir build
cd build

# Generate the makefiles and build the library
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
sudo make install
```

### Step 3: Installing Logging Library
For logging purposes, we use [Google Logging (glog)](https://github.com/google/glog). To install it:

```bash
# Clone the glog repository
git clone https://github.com/google/glog
cd glog

# Create build directory
mkdir build
cd build

# Generate the makefiles with necessary options disabled
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF ..
make -j8
sudo make install
```
### Step 4: Installing Tokenization Library
[SentencePiece](https://github.com/google/sentencepiece) is used for text tokenization, especially for handling multilingual input.

```bash
# Clone the SentencePiece repository
git clone https://github.com/google/sentencepiece
cd sentencepiece

# Create build directory
mkdir build
cd build

# Generate the makefiles and build the library
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
sudo make install
```
### Step 5: Verifying Installation
Once all libraries are installed, you can verify their installation by cloning the project and running tests.

```bash
# Clone the LlamaInfer project
git clone https://github.com/AllenJWZhu/LlamaInfer
cd LlamaInfer
```

Navigate to the test directory and run the test for math libraries:

```bash
# Build the project using CMake and WSL toolchain (if necessary)
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=<path_to_wsl_toolchain> ..
make -j8

# Test the math functionalities
./test/test_math
```
If the tests pass without errors, the libraries have been successfully installed.

## Model Support
Currently supported models:

- LLaMA (7B, 13B, 33B, 65B)
- LLaMA2 (7B, 13B, 70B)
- TinyLLaMA

To use a custom model, export it using the provided script:
```
python export_model.py --model llama2_7b --output llama2_7b.bin --meta-llama path/to/llama/model/7B
```

## Advanced Features
### Quantization
Enable int8 quantization to reduce memory usage:
```
bashCopy./llm_infer --model path/to/llama_7b_int8.bin --quantize int8
```

### Custom CUDA Kernels
The engine implements optimized CUDA kernels for:
- RMSNorm
- Softmax
- SwiGLU activation
- Multi-head attention

These can be found in 
```
src/cuda/kernels/.
```

### Memory-Mapped Model Loading
Large models are loaded using memory mapping, allowing for fast startup times and shared memory across multiple processes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
