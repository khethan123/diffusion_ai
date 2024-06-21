# Diffusion AI

## Overview

Welcome to `diffusion_ai`! This small library is the result of my learning journey from the [Practical Deep Learning for Coders](https://course.fast.ai/Lessons/part2.html) course. This course, created by Jeremy Howard, Jonathan Whitaker, and Tanishq Abraham, covers the Stable Diffusion algorithm and its variants from scratch. Using this course I deepend my understanding of diffusion models and even implemented them on my own. So can you by using this library.

### Predicted Images and Videos From 

original image
![original_dataset](https://github.com/khethan123/diffusion_ai/assets/100506743/7a563970-9d93-4dff-830f-bd5727d3e5b1)


Here are some predicted images and videos generated using `diffusion_ai`:

normal prediction: 
![predicted_image](https://github.com/khethan123/diffusion_ai/assets/100506743/b9045535-6c19-4822-8070-14dea12394d8)


prediction based on class
![predicted_class](https://github.com/khethan123/diffusion_ai/assets/100506743/d41c6aae-09e6-4405-b7d9-2af57ce10fb5)

ddpm denoising process

https://github.com/khethan123/diffusion_ai/assets/100506743/79975dee-110f-4809-af0f-2019d7605cce


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


https://github.com/khethan123/diffusion_ai/assets/100506743/91376e88-b506-4991-8fcc-54d1efc45241


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

If you use this library or any part of the course content, please cite:

- Howard, J., Whitaker, J., & Abraham, T. (2024). Practical Deep Learning for Coders. Retrieved from [Practical Deep Learning for Coders](https://course.fast.ai/Lessons/part2.html).

## Get Started Now!

Dive into the exciting world of diffusion models and deep learning by getting started with `diffusion_ai` today. Happy coding!


Yes, you can use the content from the course if it is licensed under the Apache 2.0 License. The Apache 2.0 License is permissive and allows you to use, modify, and distribute the content, even for commercial purposes, as long as you comply with the terms of the license.

Here are some key points to keep in mind:

1. **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. This can be done in your README file.
2. **Notice**: You must include a copy of the Apache 2.0 License in any distribution of the work.
3. **Changes**: If you modified the content, you must provide a notice of the changes.

Here's how you can include the required information in your README:

### Citations

This project was made possible by the "Practical Deep Learning for Coders" course by Jeremy Howard, Jonathan Whitaker, and Tanishq Abraham. Their teachings provided the foundation for this library.

**Citation**:
Howard, Jeremy, Jonathan Whitaker, and Tanishq Abraham. "Practical Deep Learning for Coders." *course.fast.ai*, 2024, https://course.fast.ai/. Accessed 21 June 2024.

### License

This project incorporates content licensed under the Apache 2.0 License from the "Practical Deep Learning for Coders" course. The full text of the Apache 2.0 License is included below.

```plaintext
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.
      ...
```

You can include the full text of the Apache 2.0 License in a `LICENSE` file in your repository. Here’s a link to the full text: [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0).

In addition to your README, you should also include a `LICENSE` file in your repository that contains the full text of the Apache 2.0 License. Here is an example `LICENSE` file:

```plaintext
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

...

END OF TERMS AND CONDITIONS
```

By following these steps, you can use the content from the course while complying with the Apache 2.0 License.


---

For any questions or contributions, please feel free to open an issue or pull request.
