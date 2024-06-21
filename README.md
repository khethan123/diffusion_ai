# Diffusion AI

## Overview

Welcome to `diffusion_ai`! This small library is the result of my learning journey from the [Practical Deep Learning for Coders](https://course.fast.ai/Lessons/part2.html) course. This course, created by Jeremy Howard, Jonathan Whitaker, and Tanishq Abraham, covers the Stable Diffusion algorithm and its variants from scratch. Using this course, I deepened my understanding of diffusion models and even implemented them on my own. You can do the same by using this library.

## Predicted Images and Videos
<p align="center">
    Original image
    <br>
    <img src="https://github.com/khethan123/diffusion_ai/assets/100506743/7a563970-9d93-4dff-830f-bd5727d3e5b1" alt="original_dataset">
</p>
<p align="center">
    Here are some predicted images and videos generated using `diffusion_ai`:
    <br>
    Uncoditional Generation
    <br>
    <img src="https://github.com/khethan123/diffusion_ai/assets/100506743/b9045535-6c19-4822-8070-14dea12394d8" alt="predicted_image">
</p>
<br>
<p align="center">
    Class Conditioned generation
    <br>
    <img src="https://github.com/khethan123/diffusion_ai/assets/100506743/d41c6aae-09e6-4405-b7d9-2af57ce10fb5" alt="predicted_class">
</p>
<br>
<div align="center">
    DDPM denoising process
    <br>
    <video src="https://github.com/khethan123/diffusion_ai/assets/100506743/79975dee-110f-4809-af0f-2019d7605cce" controls>
</div>

## Getting Started

To run `diffusion_ai`, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/khethan123/diffusion_ai.git
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

- **FID Score**: 4.058 on generated data
- **Model**: Unet with timestep embeddings, attention block, and class conditioning
- **Training Loss**: 3.2%
- **Evaluation Loss**: 3.3%
- **Training Duration**: 25 epochs
- **Sampler**: DDIM
- **Image Size**: 32x32 pixels
- **Time Steps**: 100

## Get Started Now!

Dive into the exciting world of diffusion models and deep learning by getting started with `diffusion_ai` today. Happy coding!

## Acknowledgements

This project was made possible by the "Practical Deep Learning for Coders" course by Jeremy Howard, Jonathan Whitaker, and Tanishq Abraham, PhD. Their teachings provided the foundation for this library.

## Citations

This project was made possible by the "Practical Deep Learning for Coders" course by [course.fast.ai](https://course.fast.ai)
If you use this library or any part of the course content, please cite:

**Citation**:
Jeremy Howard, Jonathan Whitaker, and Tanishq Abraham. "Practical Deep Learning for Coders." [*course.fast.ai*, 2022, part 2](https://course.fast.ai/.) Accessed 21 June 2024.
See CITATION file for more details.

### License

This project incorporates content licensed under the Apache 2.0 License from the "Practical Deep Learning for Coders" course.
The full text of the Aphache 2.0 License is present in the repository.

---

For any questions or contributions, please feel free to open an issue or pull request.
