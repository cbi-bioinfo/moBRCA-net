# BRCAnet: a Breast Cancer Subtype Classification Framework Based on Multi-Omics Attention Neural Networks
BRCAnet is an omics-level attention-based breast cancer subtype classification framework that uses multi-omics datasets. Dataset integration was performed based on feature-selection modules that consider the biological relationship between the omics datasets (gene expression, DNA methylation, and microRNA expression). Moreover, for omics-level feature importance learning, a self-attention module was applied for each omics feature, and each feature was then transformed to the new representation incorporating its relative importance for the classification task. The representation of each omics dataset was concatenated and delivered to the fully connected layers to predict the breast cancer subtype of each patient.

## Requirements
* Tensorflow (>= 1.8.0)
* Python (>= 2.7)
* Python packages : numpy, pandas
