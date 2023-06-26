# Learn Artiphysiology - Neurophysiology Applied to Artificially Intelligent Systems

'Artiphysiology' is a term coined by Dean A. Pospisil, Anitha Pasupathy, and Wyeth Bair in their paper ['Artiphysiology' reveals V4-like shape tuning in a deep network trained for image classification
(2018)](https://elifesciences.org/articles/38242). In this paper, they use this method to test whether two properties of shape selectivity in V4, tuning for boundary curvature, and translation invariance arise within a convolutional neural network (CNN) known as AlexNet. This approach is akin to how an electrophysiologist characterizes single neurons in the brain.

I have been applying Artiphysiology to examine border ownership selectivity [(Term paper)](https://drive.google.com/file/d/11tImWjiXW9stfrepN8cBhX53kX1RwTpG/view?usp=share_link) and receptive field mapping in CNNs [(Master Thesis)](https://drive.google.com/file/d/1MCTaYwBLd1Bgp-cfZlkt8A91-HsQCUmg/view?usp=share_link). Here are some lessons I have learned from my research. I will continue to add more research tips in the future.


## Setting Up
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

* As your project grows, you might want to write some specialized functions that you use repetitively in Python scripts. Here is an example: [RF-Mapping Repo](https://github.com/tonyfu97/RF-Mapping/tree/main/src/rf_mapping). I use Visual Studio Code as my IDE.
* **When working with a significant amount of data**, I would not recommend storing all the data and results on your local machine. Instead, consider using an external SSD (like a Samsung T7) of at least 256 GB (depending on your project) and at least USB 3.0. This way, when reading data, you can directly access it from the external SSD, and generate results written directly to the external SSD. For reference, I have generated about 300 GB of data for a single project. However, if you're just starting out, there's no need to invest in such a solution right away.
* However, keep your source code on your computer, and better yet, track it using GitHub. 
* The 50,000 ImageNet dataset that Dr. Pospisil used is linked [here](http://wartburg.biostr.washington.edu/loc/course/artiphys/data/i50k.html). It will consume about 60 GB, so I recommend acquiring an external SSD first if storage is a concern. The natural images are mainly used to detemine what image patches in the natural image dataset that drives the neurons the most, and this work has already been done by us (see [CNN-Database](https://github.com/tonyfu97/CNN-Database)), so you don't need to download this dataset unless you want to use it for other purposes.


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
* Familiarize yourself with these three CNNs by reading their papers: AlexNet [(Krizhevsky et al., 2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), VGGNets [(Simonyan and Andrew Zisserman, 2014)](https://arxiv.org/abs/1409.1556), and ResNets [(He et al., 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). Do not get too caught up in the details. A general understanding, e.g., what a pooling layer is or what ReLU is, will suffice. Take note that PyTorch's version of AlexNet is not the original version that won the ImageNet competition in 2012. Instead, it is a variant that was published in [2014](https://arxiv.org/abs/1404.5997).
* Be aware that the term "convolution" is a misnomer. The networks are actually performing a cross-correlation operation, as they never flip the kernel, even when the term is initially introduced in [LeCun et al., (1989)](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf).
* Understand that most of the units expect the input dataset to be standardized. The 50,000 ImageNet dataset linked above is not standardized. By standardize, I don't mean standardizing each image individually. I mean you need to compute the average and standard deviation of every pixel in the dataset, and standardize each image according to this global average and standard deviation, so that the dataset globally has a mean of 0 and a standard deviation of 1.
* It would be useful to know how to calculate the receptive field of a unit in a given layer by hand. The PyTorch official documentation (e.g., [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)) provides some formulas. I have written a script [spatial_utils.py](spatial_utils.py) that handles these calculations.
* Always make sure that you set the network to inference mode using `model.eval()`. For AlexNet, this won't cause any difference. However, for networks with special layers that change characteristics depending on whether they are in training or inference mode, such as the `BatchNorm2d` layer in ResNet18, failing to set it to `eval()` mode could result in additional normalization of the image. This could lead to inaccurate interpretations of the response.


## Data Analysis
It is important to store results in a way that can be easily accessed. As mentioned above, it's better to store these results on an external SSD rather than on your local machine. The format to save data really depends on the case and your personal preference. I generally follow these rules:


1. If the data is a dataframe, I will save it as `.csv` or `.txt`, so it can be easily loaded using `pd.read_csv()` or even queried using SQLite3.
2. If the data is image data that will be used later, I save it as `.npy` using `np.save()` and can easily load it again using `np.load()`.
3. Usually, in addition to the `.npy` file, I convert each image into a .png so I can later view it using a simple web portal (like this [CNN-Database](https://github.com/tonyfu97/CNN-Database)). You don't need to deploy it on the web like I did; it can be a local website. Since the images are saved as uint8, they also take up less space. Here is how to convert image array into .png file:

```
import numpy as np
from PIL import Image
array = array.astype(np.uint8)
image = Image.fromarray(array)
image.save("filename.png")
```
I also like to convert any DataFrame data into JSON format for accessibility in JavaScript. You can do this by first placing the data in a Python dictionary and then dumping it into a JSON file. Here's an example:
```
import json
import pandas as pd

output_dict = df.to_dict()

with open("output_file.json", 'w') as f:
    json.dump(output_dict, f)
```

## Computing Resources

If you have access to one of our lab computers, you can use the following steps to access it. The instructions provided below are specific to macOS users. For Windows users, a software named PuTTY can be used to perform similar operations.

### Setting Up
* If you don't already have an SSH key pair (consisting of a public and a private key), you'll need to create one. In Terminal, type `ssh-keygen`. Follow the prompts to create a new key pair. If you already have an SSH key pair, they are likely stored in the `~/.ssh/` directory.
* Use the `ssh-copy-id` command to copy your public key to the remote server, replacing `username` and `hostIP` with your information.
```
ssh-copy-id username@hostIP
```
and use the `logout` command to log out.
* Now, you should be able to SSH into your remote server by typing:
```
ssh username@hostIP
```
* Edit your `/etc/hosts` file to assign an alias to the IP address of the lab server. The format should be:
```
hostIP	computer_name.department_name.school_domain		alias_name
```
Now you can log in using:
```
ssh username@alias_name
```

### Running Scripts
* To run a Python script in the background even after you close the terminal window, use:
```
nohup python3 -m src.script_name &
```
Here, `nohup` allows the process to continue running even after the terminal is closed, `python3 -m` executes the named script, and `&` runs the command in the background.
* The publicly accessible server might not be powerful enough to run scripts, so you need to SSH into another computer.
* If you anticipate your script running for days, notify other lab members beforehand. Also, use the `top` command to check if other scripts are currently running.
* You can use the `lspci` command to check for an NVIDIA GPU as follows:
```
lspci | grep -i nvidia
```
If an NVIDIA GPU is installed on the machine, this command will output its details.

