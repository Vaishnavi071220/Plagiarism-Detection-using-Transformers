# Plagiarism Detection using Transformers

## Overview
Plagiarism detection is a critical challenge in academic and professional settings. This project leverages state-of-the-art transformer-based models to accurately detect plagiarized content, including verbatim copying, paraphrasing, and semantic rewording. The models evaluated include BERT, RoBERTa, T5, and a hybrid of BERT and LSTM, all trained and fine-tuned on plagiarism-specific datasets.

## Authors
- Vaishnavi Kukkala (vkukk2@unh.newhaven.edu)
- Hari Krishna Para (hpara2@unh.newhaven.edu)
- Sai Charan Chandu Patla (schan30@unh.newhaven.edu)

University of New Haven, Department of Data Science

## Key Results

| Model               | Accuracy (D1) | Accuracy (D2) |
|---------------------|---------------|---------------|
| BERT                | 85.87%        | 83.77%        |
| BERT+LSTM Hybrid    | 85.20%        | 82.84%        |
| RoBERTa             | 84.46%        | 82.03%        |
| T5                  | 77.84%        | 66.49%        |

## Features
- **Transformer-Based Models**: Fine-tuned versions of BERT, RoBERTa, and T5 for plagiarism detection.
- **Hybrid Architectures**: Combines BERT's contextual embeddings with LSTM for sequential dependency modeling.
- **Multi-Dataset Testing**: Evaluation on SNLI (Dataset 1) and MRPC (Dataset 2).
- **Preprocessing Pipeline**: Efficient tokenization, padding, truncation, and special token handling.

## Methodology

### Datasets
- **SNLI (Stanford Natural Language Inference)**: Modified to detect semantic relationships indicative of plagiarism.
- **MRPC (Microsoft Research Paraphrase Corpus)**: Focused on sentence paraphrasing.

### Preprocessing Steps
- **Tokenization**: Model-specific tokenizers (BERT, RoBERTa, T5).
- **Special Tokens**: Sequence structure tokens added.
- **Padding and Truncation**: Standardized input lengths for batch processing.
- **Vectorization**: Embedding vectors generated from tokenized text.

### Models
- **BERT**: Bidirectional Transformer for semantic understanding.
- **RoBERTa**: Optimized version of BERT with enhanced training.
- **BERT+LSTM Hybrid**: Combines BERT embeddings with LSTM's sequential analysis.
- **T5**: Treats tasks as text-to-text problems for flexibility.

### Training Configuration
- **Loss Function**: Binary cross-entropy for classification tasks.
- **Batch Size**: 16 (optimized for GPU memory).
- **Epochs**: 5 (selected to avoid overfitting).
- **Metrics**: Accuracy, precision, recall, and F1-score.

## How to Use

### 1. Access the Notebook
Download or clone this repository and open the notebook file `Plagiarism_Detection_code.ipynb`.

### 2. Setup Environment
Run the notebook on Google Colab or a Jupyter Notebook locally. Ensure you have Python 3.12 installed if working locally.

### 3. Dataset Preparation
Download the datasets (SNLI and MRPC) provided in the code using the Hugging Face datasets library. Alternatively, add a shortcut of your custom dataset folder to your Google Drive for easy access.

### 4. Run the Notebook
Execute the cells in the notebook sequentially. The notebook covers:
- Installation of required dependencies (transformers, datasets, pandas).
- Loading and preprocessing datasets.
- Training and evaluation of transformer models for plagiarism detection.

### 5. Customize
You can modify hyperparameters such as batch size, learning rate, etc. Experiment with additional models or datasets for comparative analysis.

### 6. Save Results
Evaluation metrics such as accuracy, precision, recall, and F1-score will be displayed during training. You can save model checkpoints and desired outputs.

## Results
- **Best Overall Model**: BERT, achieving top accuracy and F1 scores on both datasets.
- **Hybrid Model**: Demonstrated robust performance in detecting sequential patterns.
- **T5**: Struggled with nuanced paraphrasing on MRPC dataset.

## Conclusion and Future Work
This project demonstrates the potential of transformers in plagiarism detection, with BERT leading the results. Future improvements may include:
- **Domain-Specific Models**: Fine-tuning on specialized plagiarism datasets.
- **Real-Time Detection**: Developing scalable, real-time plagiarism detection systems.
- **Multilingual Support**: Extending the model's capabilities to work with multiple languages.

### Limitations
- **High Computational Costs**: Transformer models require significant computational resources.
- **Generalization**: Domain-specific and multilingual adaptations remain challenging.

## References
- S. V. Moravvej et al., "A Novel Plagiarism Detection Approach Combining BERT-based Word Embedding..."
- R. Patil et al., "A Novel Natural Language Processing Based Model for Plagiarism Detection."
