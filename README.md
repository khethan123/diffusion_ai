# Diffusion AI

## Overview

Welcome to `diffusion_ai`! This small library is the result of my learning journey from the "Practical Deep Learning for Coders" course. This course, created by Jeremy Howard, Jonathan Whitaker, and Tanishq Abraham, PhD, covers the Stable Diffusion algorithm and its variants from scratch. It has over 30 hours of video content and provides rigorous coverage of the latest techniques in deep learning.

### Course Highlights

In the course, you will:
- Implement Stable Diffusion and other diffusion models like DDPM and DDIM.
- Explore deep learning topics such as neural network architectures, data augmentation, and various loss functions.
- Build models from scratch including MLPs, ResNets, Unets, autoencoders, and transformers.
- Use PyTorch to implement models and create a deep learning framework called miniai.
- Master Python concepts to keep your code clean and efficient.
- Apply fundamental concepts like tensors, calculus, and pseudo-random number generation to machine learning techniques.

For more details about the course, visit [Practical Deep Learning for Coders](https://course.fast.ai/Lessons/part2.html).

### Predicted Images and Videos

Here are some predicted images and videos generated using `diffusion_ai`:

**[Placeholder for Images]**

**[Placeholder for Videos]**

## Getting Started

To run `diffusion_ai`, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd diffusion_ai
   ```

2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Follow the Jupyter Notebook Tutorials**:
   Explore the tutorials to implement DDPM, DDIM, Unet, timestep embeddings, class-conditioned outputs, and training with VAE latents.

## Example Results

### Performance on Fashion MNIST Dataset

- **FID Score**: 4.058064770194278 on generated data
- **Model**: Unet with timestep embeddings, attention block, and class conditioning
- **Training Loss**: 3.2%
- **Evaluation Loss**: 3.3%
- **Training Duration**: 25 epochs
- **Sampler**: DDIM
- **Image Size**: 32x32 pixels
- **Time Steps**: 100

## Course Content Summary

### Diffusion Foundations
- **DDPM and DDIM**: Implement noise prediction models, visualize noisy images, and explore noise schedules.
- **Samplers**: Experiment with Euler sampler, Ancestral Euler sampler, Heuns method, LMS sampler.
- **Stable Diffusion Models**: Implement unconditional and conditional models, and solve inverse problems.

### Advanced Topics
- **Textual Inversion and Dreambooth**: Create unique models and explore Hugging Face’s Diffusers library.
- **Deep Learning Optimizers**: Learn about SGD, Momentum, RMSProp, Adam, and learning rate schedulers.
- **Python Concepts**: Use iterators, generators, dunder methods, decorators, and more for efficient code.

### Basic Foundations
- **Tensors and Calculus**: Understand matrix multiplication, derivatives, and loss functions.
- **Neural Network Architectures**: Build MLPs, ResNets, autoencoders, Unets, and transformers.
- **Deep Learning Techniques**: Data augmentation, dropout, mixed precision training, and normalization.

### Machine Learning Techniques and Tools
- **Clustering and CNNs**: Implement mean shift clustering and build CNNs from scratch.
- **Experiment Tracking**: Use Weights and Biases (W&B) for tracking experiments and metrics.

## Acknowledgements

This project was made possible by the "Practical Deep Learning for Coders" course by Jeremy Howard, Jonathan Whitaker, and Tanishq Abraham, PhD. Their teachings provided the foundation for this library.

## Citations
```ABA
If you use this library or any part of the course content, please cite:

- Howard, J., Whitaker, J., & Abraham, T. (2024). Practical Deep Learning for Coders. Retrieved from [Practical Deep Learning for Coders](https://course.fast.ai/Lessons/part2.html).
```
## Get Started Now!

Dive into the exciting world of diffusion models and deep learning by getting started with `diffusion_ai` today. Happy coding!

---

For any questions or contributions, please feel free to open an issue or pull request.
