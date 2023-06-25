# Learn Artiphysiology - Neurophysiology Applied to Artificially Intelligent Systems

'Artiphysiology' is a term coined by Dean A. Pospisil, Anitha Pasupathy, and Wyeth Bair in their paper ['Artiphysiology' reveals V4-like shape tuning in a deep network trained for image classification
(2018)](https://elifesciences.org/articles/38242). In this paper, they use this method to test whether two properties of shape selectivity in V4, tuning for boundary curvature, and translation invariance arise within a convolutional neural network (CNN) known as AlexNet. This approach is akin to how an electrophysiologist characterizes single neurons in the brain.

I have been applying Artiphysiology to examine border ownership selectivity [(Zhou et al., 2000)](https://www.jneurosci.org/content/20/17/6594) and receptive field mapping in CNNs. Here are some lessons I have learned from my research. I will continue to add more research tips in the future.


## Setting up
* Colab notebooks are a good starting point, especially for exploring ideas. I recommend beginning with a small project using notebooks.
* Download the following Python packages:
```
python==3.9.7
matplotlib==3.4.3
numpy==1.20.3
opencv-python==4.7.0.72
pandas==1.3.4
Pillow==8.4.0
scipy==1.10.0
torch==1.11.0
torchvision==0.12.0
tqdm==4.62.3
```

You can download them with the following commands:
* On macOS, use the terminal:
```
pip install python==3.9.7 matplotlib==3.4.3 numpy==1.20.3 opencv-python==4.7.0.72 pandas==1.3.4 Pillow==8.4.0 scipy==1.10.0 torch==1.11.0 torchvision==0.12.0 tqdm==4.62.3
```

* On Windows, use cmd:
```
pip install python==3.9.7 matplotlib==3.4.3 numpy==1.20.3 opencv-python==4.7.0.72 pandas==1.3.4 Pillow==8.4.0 scipy==1.10.0 torch==1.11.0 torchvision==0.12.0 tqdm==4.62.3
```

The versions do not have to match exactly, but please note that PyTorch has deprecated a feature that I often use (i.e., 'hooks').

* As your project grows, you might want to write some specialized functions that you use repetitively in Python scripts. Here is an example: [RF-Mapping-Repo](https://github.com/tonyfu97/RF-Mapping/tree/main/src/rf_mapping). 
* When working with a significant amount of data, I would not recommend storing all the data and results on your local machine. Instead, consider using an external SSD (like a Samsung T7) of at least 256 GB (depending on your project) and at least USB 3.0. This way, when reading data, you can directly access it from the external SSD, and generate results written directly to the external SSD. Keep your source code locally, and better yet, track it using GitHub. For reference, I have generated about 300 GB of data for a single project.
* The 50,000 ImageNet dataset that Dr. Pospisil used is linked [here](http://wartburg.biostr.washington.edu/loc/course/artiphys/data/i50k.html). It will consume about 60 GB, so I recommend acquiring an external SSD first if storage is a concern.


## Artiphysiogy
Here is an [example notebook](rf_mapping_live(standalone).ipynb) that demonstrates Artiphysiology in action [(demo video)](https://youtu.be/Xc0pfPmdJcY). It is used to create an animation of how a color bar map is created. In the **Helper functions and classes** section, please pay attention to:

* How the bar is created.
* The reasoning behind truncating the model up to the layer of interest before running any inference.
* What is a hook function? Note: PyTorch's hooks are deprecated recently. If you download the most recent version, some of the code might not work.
* What is 'xn'? How is it different from the size of the receptive field? Why do you need to determine 'xn'? If you are wondering about the 'xn' and RF size of different layers in AlexNet, VGG16, and ResNet18, check out [model_info.txt](model_info.txt).
* Why we extract only the unit at the spatial center? Dr. Pospisil provides a good explanation: "AlexNet contains over 1.5 million units organized in eight major layers, but its convolutional architecture means that the vast majority of those units are spatially offset copies of each other. For example, in the first convolutional layer, Conv1, there are only 96 distinct kernels, but they are repeated everywhere on a 55 × 55 grid. Thus, for the convolutional layers, Conv1 to Conv5, it suffices to study the selectivity of only those units at the spatial center of each layer." Note that he refers to the Caffe AlexNet, not the PyTorch version we are using.
* How are the bar maps made? Currently, the bar map is a weighted average of the bar. This is why we need to keep the background of the bar zero, rather than using negative numbers, to prevent bars from cancelling each other out during addition.
* Have some fun with [Receptive Field Playground](https://github.com/tonyfu97/rf_playground). It is an interactive web app that allows you to map RF of CNNs using bar stimulus on your browser.


## CNN Basics
* Familiarize yourself with these three CNNs by reading their papers: AlexNet [(Krizhevsky et al., 2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), VGGNets [(Simonyan and Andrew Zisserman, 2014)](https://arxiv.org/abs/1409.1556), and ResNets [(He et al., 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). Do not get too caught up in the details. A general understanding, e.g., what a pooling layer is or what ReLU is, will suffice.
* Be aware that the term "convolution" is a misnomer. The networks are actually performing a cross-correlation operation, as they never flip the kernel, even when the term is initially introduced in [LeCun et al., (1989)](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf).
* Understand that most of the units expect the input dataset to be standardized. The 50,000 ImageNet dataset linked above is not standardized. By standardize, I don't mean standardizing each image individually. I mean you need to compute the average and standard deviation of every pixel in the dataset, and normalize each image according to this global average and standard deviation, so that the dataset globally has a mean of 0 and a standard deviation of 1.
* It would be useful to know how to calculate the receptive field of a unit in a given layer by hand. The PyTorch official documentation (e.g., [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)) provides some formulas. I have written a script [spatial_utils.py](spatial_utils.py) that handles these calculations.


## Data Analysis
It is important to store results in a way that can be easily accessed. As mentioned above, it's better to store these results on an external SSD rather than on your local machine. The format to save data really depends on the case and your personal preference. I generally follow these rules:


1. If the data is a dataframe, I will save it as .csv or .txt, so it can be easily loaded using pd.read_csv() or even queried using SQLite3.
2. If the data is image data that will be used later, I save it as .npy using np.save() and can easily load it again using np.load().
3. Usually, in addition to the .npy file, I convert each image into a .png so I can later view it using a simple web portal (like this [CNN-Database](https://github.com/tonyfu97/CNN-Database)). You don't need to deploy it on the web like I did; it can be a local website. Since the images are saved as uint8, they also take up less space. Here is how to convert image array into .png file:

```
import numpy as np
from PIL import Image
array = array.astype(np.uint8)
image = Image.fromarray(array)
image.save("filename.png")
```
