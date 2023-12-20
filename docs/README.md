# VVAD-LRS3
Library to provide models trained on the VVAD-LRS3 Dataset. The library also contains preprocessing pipelines.
Applications are Speaker detection in scenarios, where multiple people are in the robot's field of view 
and stare detection for proactive approaches.

# Important Links
|Topic      |Content        |
|-----------|---------------|
|[**Docs**](https://adriandavidauer.github.io/VVAD/)|Code and Feature documentation|
|[**Paper**](https://doi.org/10.48550/arXiv.2109.13789)|Scientific publication|

# Prerequisites
vvadlrs3 depends on dlib which needs build tools to be installed over pip.
[Here](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/) is described what is needed.

On Ubuntu, you need to install the following:

```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

# Install
```bash
pip install vvadlrs3
```

# Data
The models are trained on the [VVAD-LRS3](https://www.kaggle.com/datasets/adrianlubitz/vvadlrs3) Dataset.

<p align="center">
    <img src="sampleVisualization.gif">
    <br>
    <sup>Some samples visualized. Samples with green borders are positive samples, samples with red borders are negative samples</sup>
</p>


# How to contribute
Please check out the [Contributing Page](./how-to-contribute.md)