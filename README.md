# Project Summary: Google Landmark Recognition 2021


## Abstract
This project addresses the challenge of fine-grained instance-level recognition using the [Google Landmarks Dataset v2 (GLDv2)](https://www.kaggle.com/competitions/landmark-recognition-2021). The primary difficulty presented by this dataset is the extreme "long-tail" distribution, where the majority of the 81,000+ landmark classes contain very few training examples, making standard classification computationally expensive and prone to overfitting.  

To investigate the feasibility of Deep Convolutional Neural Networks (CNNs) under these data-constrained environments, this solution implements a **ResNet (Residual Network)** architecture. The study focuses on a curated subset of **100 landmark classes** to benchmark the model's ability to distinguish subtle architectural features with limited supervision (averaging ~20 images per class).  

---

## Methodology
The problem was solved using a three-stage pipeline: Data Curation, Architecture Adaptation, and Supervised Training.
### 1. Data Curation & Preprocessing
Given the massive scale of the original GLDv2 dataset (5M+ images), a subset strategy was employed to make the training computationally tractable while retaining the challenge of fine-grained classification.  
* **Class Selection:** The dataset was filtered to retain the top 100 classes. This reduces the search space while maintaining high intra-class variance.  
* **Preprocessing:** Input images were resized to standard dimensions ($224 \times 224$) and normalized using ImageNet mean and standard deviation values to align with the pre-trained backbone.  
* **Imbalance Handling:** Due to the low shot count per class (~20 images), stratifying the train/validation split was crucial to ensure every class was represented in the evaluation set.  

### 2. Model Architecture: Residual Networks (ResNet)
To solve the vanishing gradient problem inherent in deep networks, this project utilizes a **ResNet** backbone. ResNet introduces "skip connections" (or shortcut connections) that allow gradients to flow through the network more easily during backpropagation.

* **Transfer Learning:** Instead of training from scratch, the model was initialized with weights pre-trained on **ImageNet**. This allows the model to leverage robust low-level feature extractors (edges, textures, shapes) learned from millions of generic images.
* **Classification Head:** The final fully connected layer (FC) of the ResNet was replaced with two dense layers with dropout to output a vector of size $N=100$, corresponding to our specific landmark classes.  


### 3. Training Strategy
The model was trained using the Cross-Entropy Loss function, which is standard for multi-class classification problems.
* **Optimization:** The weights were updated using ADAM optimizer.
* **Regularization:** To prevent the model from memorizing the small training set (overfitting), data augmentation techniques (random cropping, horizontal flipping) were applied during training to artificially increase the diversity of the dataset.  

---

## Framework & Attribution
* **Framework:** TensorFlow/Keras
* **Base Architecture:** ResNet152 (Pre-trained on ImageNet)
* **Training:** Training was done on Kaggle. The code is available in Kaggle (https://www.kaggle.com/code/mohammadkhanfau/notebookglr-v1). 
* **Dataset Citation:** T. Weyand, A. Araujo, B. Cao and J. Sim, "Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval," *Proc. CVPR*, 2020. 
* **Code References:**
- https://www.kaggle.com/code/mohitsinghdhaka/exercise3-notebook-glr2021
- https://www.kaggle.com/code/takedarts/inference-and-submission-pytorch-resnet34
- https://www.kaggle.com/code/lucamtb/glrecogn21-efficientnetb0-baseline-inference




---

## Results
A comprehensive visual analysis of the model's performance is located in the `results/` directory. Due to the high difficulty of the dataset, evaluation focuses on the model's convergence and qualitative prediction capabilities rather than raw top-1 accuracy.

The `results` folder contains:
1.  **Training Metrics:** Graphs displaying the Training and Validation Loss over epochs. The declining loss curve demonstrates that the ResNet backbone successfully adapted its feature maps to the specific domain of landmark recognition.
2.  **Sample Predictions:** A visual collage of inference results on the validation set, highlighting the model's ability to correctly identify distinct architectural features of famous landmarks despite the limited training data.

