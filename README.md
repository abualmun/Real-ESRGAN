# 🛂 Passport Photo Enhancer with Real-ESRGAN

This project fine-tunes the powerful [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) model to enhance **passport and ID-style facial images**, focusing on improving clarity and realism for low-quality or compressed photos.

<p align="center">
  <img src="https://user-images.githubusercontent.com/13007665/131244992-eda96b33-4c83-4d00-9584-4e44e2f36c83.png" width="500"/>
</p>

---

## 📦 Model Overview

**Base Model:** Real-ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)

**Modifications:**
- Fine-tuned on passport-like face crops from **Flickr-Faces-HQ (FFHQ)** dataset
- Emphasis on recovering fine facial details such as eyes, skin texture, and edges under compression and blurring

**Target Use Case:**
- Governmental or verification systems
- Enhancing historical or scanned passport images
- Identity restoration and enhancement pipelines

---

## 🧠 Training Details

- **Dataset:** [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq)
  - 70,000 high-quality face images
  - Preprocessed to simulate passport-like degradation (e.g., compression, resizing, blur)

- **Training Platform:** Google Colab
  - Used TPU/High-RAM runtime
  - Mixed precision training with PyTorch + CUDA
  - Custom augmentations and passport-style degradations applied on-the-fly

- **Loss Functions:**
  - Pixel Loss (L1)
  - Perceptual Loss (VGG-based)
  - Adversarial Loss (GAN Discriminator)
  - Identity Loss (optional; based on ArcFace embeddings)

---

## 🚀 How to Use

1. Clone the repository and install dependencies
2. Upload your passport photo to the `inputs/` folder
3. Run the inference script:

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus_passport -i inputs --outscale 4
