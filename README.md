# BRCAnet: a Breast Cancer Subtype Classification Framework Based on Multi-Omics Attention Neural Networks
BRCAnet is an omics-level attention-based breast cancer subtype classification framework that uses multi-omics datasets. Dataset integration was performed based on feature-selection modules that consider the biological relationship between the omics datasets (gene expression, DNA methylation, and microRNA expression). Moreover, for omics-level feature importance learning, a self-attention module was applied for each omics feature, and each feature was then transformed to the new representation incorporating its relative importance for the classification task. The representation of each omics dataset was concatenated and delivered to the fully connected layers to predict the breast cancer subtype of each patient.

![Figure](https://github.com/cbi-bioinfo/BRCAnet/blob/main/fig1_v7.png?raw=true)

## Requirements
* Tensorflow (>= 1.8.0)
* Python (>= 2.7)
* Python packages : numpy, pandas

## Usage
Clone the repository or download source code files and prepare breast cancer multi-omics dataset including gene expression, DNA methylation, and microRNA expression.

1. Edit **"run_BRCAnet.sh"** file having multi-omics dataset files for model training and testing with subtype label for each sample. Modify each variable values in the bash file with filename for your own dataset. Each file shoudl contain the header and follow the format described as follows :

- ```train_X, test_X``` : File with a matrix or a data frame containing gene expression, DNA methylation beta value, and miRNA expression of features for model training and testing, where each row and column represent **sample** and **feature**, respectively. The order of the features should be as follows: (1) Gene, (2) CpG cluster, (3) microRNA. Example for dataset format is provided below.

```
A1BG,A1CF,...,A2ML1,cg000001,cg000002,...,cg000005,hsa-mir-1249,hsa-mir-1251,...,hsa-mir-221
0.342,0.044,...,0.112,0.894,0.342,...,0.112,0.013,0.444,...,0.234
...
```

- ```train_Y, test_Y``` : File with a matrix or a data frame contatining subtype label for each sample, where each row represent **sample**. Subtype names used for training and testing should be included and users should label each subtype as 1 and 0 for others in the same order in training dataset to be matched. Example for data format is described below.

```
LumA,LumB,Her2,Basal,Normal
1,0,0,0,0
0,0,1,0,0
0,1,0,0,0
...
```

2. Use **"run_BRCAnet.sh"** to classify subtypes in multi-omics dataset.

3. You will get an output **"prediction.csv"** with classified subtypes for test dataset, and an output **"attn_score_eachOmics.csv"** with the relative importance values (attention scores) measured by BRCAnet for each omics features.


## Contact
If you have any question or problem, please send an email to **miniymay AT sookmyung.ac.kr**
