# Membership_Inference_Attack_DP

Acknowledgemnet and Reference : This code relies on the research work done as https://www.biorxiv.org/content/10.1101/2020.08.03.235416v1.full. and its github https://github.com/work-hard-play-harder/DP-MIA

This repository adds an extra feature of testing privacy related ML attacks by building multi-class classificatuion models using Differential privacy.

This repository contains 2 models CNN and LSTM trained using Multi-class classification problem using DIfferential Privacy (with loss function as SparseCategoricalCrossentropy)

Requirements
------------
Python 3.5 or higher 
TensorFlow 1.14 or 1.15


Steps to execute
-----------------

1. After downloading the code , fill the path ='' with your current direectory path
2. Dataset is already present with the cloned repository , does not need download. It uses daata from http://archive.ics.uci.edu/ml/datasets/Adult
3. LSTM and CNN (conv1D) models are used to train the shadow and attack model

![alt text](https://i1.wp.com/miro.medium.com/max/3160/1*86qGRsUqBfV2wMh13OJ2Rw.png?w=525&ssl=1)

4. L1 Kernel Regularization can be used to diminish the effect of overfitting and loweringthe attack accuracy.
5. Both the models are trained with Differential privacy using DPGradientDescentGaussianOptimizer
