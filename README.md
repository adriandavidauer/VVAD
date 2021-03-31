# VVAD-LRS3
Library to provide models trained on the VVAD-LRS3 Dataset. The library also contains preprocessing pipelines.
Applications are Speaker detection in scenarios, where multiple people are in the robot's field of view 
and stare detection for proactive approaches. 

<!-- Add link to the Paper when published -->

# Prerequisites
vvadlrs3 depends on dlib which needs build tolls to be installed over pip.
[Here](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/) is described what is needed.

For Ubuntu you just need to install the following:

```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

# Install
```bash
pip install vvadlrs3
```

# Data
The models are trained on the VVAD-LRS3 Dataset

<p align="center">
    <img src="sampleVisualization.gif">
    <br>
    <sup>Some samples visualized. Samples with green borders are positive samples, samples with red borders are negative samples</sup>
</p>


