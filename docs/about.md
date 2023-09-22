Robots are becoming everyday devices, increasing their interaction with humans. To make 
human-machine interaction more natural, cognitive features like Visual Voice Activity 
Detection (VVAD), which can detect whether a person is speaking or not, given visual 
input of a camera, need to be implemented. Neural networks are state of the art for 
tasks in Image Processing, Time Series Prediction, Natural Language Processing and other
domains. Those Networks require large quantities of labeled data. Currently there are 
not many datasets for the task of VVAD. 

In this work we created a large scale dataset 
called the VVAD-LRS3 dataset, derived by automatic annotations from the LRS3 dataset. 
The VVAD-LRS3 dataset contains over 44K samples, over three times the next competitive 
dataset (WildVVAD). We evaluate different baselines on four kinds of features: facial 
and lip images, and facial and lip landmark features. With a Convolutional Neural 
Network Long Short Term Memory (CNN LSTM) on facial images an accuracy of 92% was 
reached on the test set. A study with humans showed that they reach an accuracy of 
87.93% on the test set.[^fn1]

[^fn1]:  Adrian Lubitz, "The VVAD-LRS3 Dataset for Visual Voice Activity Detection", 
<https://doi.org/10.48550/arXiv.2109.13789>, Tue, 28 Sep 2021