# deep-learning-project-4
Vision-Language Representation Learning with Transformers
Project Overview
This project explores learning shared semantic representations between images and natural language using Transformer-based models, inspired by the CLIP architecture. The goal is to embed images and text into a common vector space where semantically related pairs are close and unrelated pairs are far apart. This shared representation enables tasks such as image-text retrieval and zero-shot inference without task-specific training.

Dataset
The Flickr8k dataset is used for this project, comprising approximately 8,000 natural images, each paired with five human-written captions. Each training example consists of an image and one of its corresponding captions.

Model Architecture
The core of the model consists of two main components:

Vision Transformer (ViT) Image Encoder: Processes images by converting them into sequences of patch embeddings, incorporating positional embeddings and a learnable classification token, followed by a Transformer encoder stack.
Transformer-based Text Encoder: Processes captions by converting them into sequences of token embeddings, incorporating positional embeddings, followed by a Transformer encoder stack.
Both encoders are designed to output vectors of the same dimension, which are then projected into a shared embedding space and normalized.

Training Objective
The models are trained using a contrastive learning objective. Within a mini-batch, the model aims to maximize the cosine similarity between matching image-caption pairs and minimize the similarity between non-matching (negative) pairs. This encourages the learning of semantically aligned representations.

Self-Supervised Pretraining
To enhance visual representations, the Vision Transformer image encoder is first pretrained using an image-only self-supervised learning (SimCLR-style) objective. This stage involves generating two different augmented views for each image and training the ViT to produce similar representations for views of the same image while differentiating views of different images. This pretraining allows the model to learn meaningful visual structures from images alone, addressing limitations of relying solely on paired image-caption data.

Evaluation Metrics
The performance of the learned representations is evaluated using:

Recall@K: For both image-to-text and text-to-image retrieval, Recall@K measures the percentage of queries for which at least one correct match appears among the top K retrieved results.
Zero-Shot Caption Selection Accuracy: Given an image and a set of N candidate captions (one correct, N-1 incorrect), the model selects the caption with the highest cosine similarity to the image embedding. Accuracy is reported for various values of N.
Setup and Dependencies
To run this project, you will need to install the following Python libraries. All necessary imports are handled in the provided notebook.

# Standard Python libraries
pip install numpy matplotlib pillow tqdm
# PyTorch and torchvision
pip install torch torchvision
How to Run
Download and Extract Data: The notebook includes commands to automatically download and extract the Flickr8k dataset.
!wget -q "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
!unzip -q flickr8k.zip -d ./flickr8k
Execute Notebook Cells: Run all cells in the Jupyter/Colab notebook sequentially. The notebook will:
Set up global configurations and data preprocessing pipelines.
Build the Vocabulary and prepare Flickr8kDataset and DataLoaders.
Define and initialize the VisionTransformer, TextEncoder, and CLIPModel.
Train the CLIPModel from scratch and evaluate its performance.
Perform self-supervised pretraining on the VisionTransformer.
Train the CLIPModel again, initializing the image encoder with the SSL-pretrained weights.
Evaluate and compare the results of both training approaches (from scratch vs. with SSL pretraining) using retrieval and zero-shot caption selection metrics.
Key Findings (Summary)
Impact of Self-Supervised Pretraining: The experiments demonstrate that self-supervised pretraining significantly improves the quality of visual representations. Models initialized with a SSL-pretrained Vision Transformer generally achieve higher Recall@K values in retrieval tasks and better accuracy in zero-shot caption selection compared to models trained from scratch.
Zero-Shot Generalization: The ability of the model to perform caption selection without explicit training for this task highlights the effectiveness of contrastive learning in creating a semantically rich shared embedding space.
Low-Data Regime: For datasets like Flickr8k with relatively limited paired data, SSL pretraining provides crucial visual inductive biases that Vision Transformers (which lack strong spatial inductive biases of CNNs) benefit from greatly, leading to more robust and generalized representations.
