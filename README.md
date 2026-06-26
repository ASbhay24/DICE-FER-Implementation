# DICE-FER: Decoupling Identity Confounders for Enhanced Facial Expression Recognition 🎭

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZEXMLnACMYqn0OpVh-RgCO86aiOxnCl4?usp=sharing)
*(Click above to run the live inference demo instantly in your browser)*

## 📌 Project Overview
Facial Expression Recognition (FER) models often struggle in real-world scenarios because they unintentionally memorize identity-specific traits (like bone structure or facial features) rather than focusing purely on the expression. 

This repository contains the PyTorch implementation of the **DICE-FER** framework, developed as a course project for **EE656** at **IIT Kanpur**. It utilizes a dual-encoder adversarial network to mathematically decouple identity confounders from expression features via Mutual Information (MI) minimization.

## 🚀 Key Results
* **Convergence:** Achieved **99.99% accuracy** on the RAF-DB dataset training split.
* **Robustness:** Successfully disentangled identity from expression using an adversarial Minimax game between a ResNet-18 Expression Encoder and an MLP Discriminator.
* **In-the-wild Ready:** Integrated OpenCV Haar Cascades for automatic face detection and cropping from raw, cluttered photographs.

## 🏗️ Architecture & Methodology
The framework relies on three main components trained jointly:
1. **Expression Encoder ($E_{exp}$):** A ResNet-18 backbone trained to classify 7 basic emotions while simultaneously attempting to fool the discriminator.
2. **Identity Encoder ($E_{id}$):** A parallel ResNet-18 backbone capturing pure identity features.
3. **Discriminator ($D$):** A 3-layer MLP trained to distinguish between real `(expression, identity)` pairs and shuffled pairs. 

**Joint Loss Function:**
The network is optimized using a combination of Cross-Entropy (classification), $L_1$ Distance (anchor similarity), and Binary Cross-Entropy (adversarial push):
$$L_{total} = L_{ce} + \lambda_1 L_1 + \lambda_2 L_{adv}$$

---

## 💻 Quick Start (Inference)

You do **not** need to train the model from scratch to test it. The pre-trained `.pth` weights are hosted in the [Releases](#) tab.

### Option 1: The 1-Click Cloud Demo (Recommended)
Simply click the **"Open in Colab"** badge at the top of this page. The notebook will automatically download the required weights from the release page, configure the environment, and allow you to upload any image for instant classification.

### Option 2: Run Locally on your Machine
If you want to run the inference script on your local CPU/GPU:

**1. Clone the repository:**
```bash
git clone [https://github.com/ASbhay24/DICE-FER-Implementation.git](https://github.com/ASbhay24/DICE-FER-Implementation.git)
cd DICE-FER-Implementation
