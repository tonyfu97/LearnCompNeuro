# 'Artiphysiology' Project Ideas

Tony Fu, Summer 2023

## Introduction

'Artiphysiology' is a approach first used by Dean A. Pospisil, Anitha Pasupathy, and Wyeth Bair in their 2018 paper ['Artiphysiology' reveals V4-like shape tuning in a deep network trained for image classification](https://elifesciences.org/articles/38242). In their work, they used this method to test if AlexNet, a convolutional neural network (CNN), shows features of shape selectivity seen in the brain's area V4. This is very much like how a scientist would study individual brain cells. Here, we want to expand on their idea of 'Artiphysiology' and suggest some new research projects. I think of 'Artiphysiology' as a process with three main steps:

In Step 1 of the project, we take a close look at important studies that have taught us a lot about how we see and perceive visual stimuli. These studies must use a clear set of visual stimuli because we'll be using these same stimuli with our neural network in Step 2.

In Step 2, we check if the CNN units show the same kind of selectivity for visual stimuli that we saw in Step 1. This usually means we'll use the stimuli with a few well-known CNNs. The Bair Lab used AlexNet, VGG16, and ResNet18 by default. But in some cases, we might choose other CNNs that are better suited for the job, and might give us clearer results.

Finally, in Step 3, we try to explain any selectivity we find by taking a closer look at the units in the network. Often, we'll come across mechanisms that are already in use or that people have suggested before. This step involves exploring wider ideas, algorithms, and models that might help us understand how deep learning can mirror the principles we see in neurophysiology. We also want to see what the challenges might be in trying to link these two areas, and where the limitations might lie.

The project proposals below are just starting points - they'll likely evolve as we make progress. And we're always open to new ideas for projects or ways to improve what we're doing. So, don't hesitate to chime in with your thoughts. Thanks!

## Possible Research Questions and Objectives

### A. Texture Segregation

* 1. Neurophysiology Literature:

| Paper Title | Authors | Year | Model Organism & Visual Area | Stimulus |
|-------------|---------|------|------------------------------|----------|
| Neuronal correlates of pop-out in cat striate cortex | Kastner, Nothdurft, & Pigarev | 1997 | Cat, Striate Cortex (V1) | Visual stimuli that differ from their surroundings in color or orientation (stimuli were single lines or bars that stood out - popped out - due to their distinct color or orientation) |
| The Neurophysiology Figure-Ground Segregation Primary Visual Cortex | Lamme | 1995 | Macaque Monkey, Primary Visual Cortex (V1) | Stimuli where figure and the background share the same orientation but differ in texture (a small central square filled with a grid of oriented bars, surrounded by a larger field of similarly oriented bars) |

* 2. Possible Deep Learning Architectures:

Besides the CNNs we already used (AlexNet, VGGNets, and ResNets), we can consider architectures that are designed for object localization and segmentation, which inherently involve some degree of figure-ground segregation.

| Paper Title | Authors | Year | Neural Network | Why relevant? |
|-------------|---------|------|----------------|----------------|
| Rich feature hierarchies for accurate object detection and semantic segmentation | Girshick, Donahue, Darrell, & Malik | 2014 | R-CNN | This network introduced the region proposal concept in deep learning for object detection. |
| Fast R-CNN | Girshick | 2015 | Fast R-CNN | Improved speed and efficiency over R-CNN by sharing computation over an entire image. |
| Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | Ren, He, Girshick, & Sun | 2016 | Faster R-CNN | Introduced the Region Proposal Network (RPN) to generate high-quality region proposals, which was a bottleneck for R-CNN and Fast R-CNN. |

* 3. Mechanism Study and Interdicipline Review:  

| Paper Title | Authors | Year | Model/Algorithm/Idea | Why relevant? |
|-------------|---------|------|---------------------|----------------|
| Gestalt psychology | Köhler | 1947 | Thingness | Introduced the concept of 'thingness' in perception, i.e., how we perceive objects as distinct entities. |
| A Century of Gestalt Psychology in Visual Perception I. Perceptual Grouping and Figure-Ground Organization | Wagemans et al. | 2012 | Gestalt Grouping Principles | Provided an overview of a century of research in Gestalt psychology, emphasizing on perceptual grouping and figure-ground organization. |
| What is an object? | Alexe, Deselaers, & Ferrari | 2010 | Objectness measure | Introduced a measure to quantify how likely it is for an image window to contain an object of any class. |
| Edge boxes: Locating object proposals from edges | Zitnick & Dollar | 2014 | EdgeBoxes | Proposed a method to create bounding box proposals for objects using edges detected in the image. |

* 4. Potential Challenges and Limitations: 

Region proposal methods, in essence, simplify the complex task of identifying and delineating objects within an image by focusing on bounding boxes that are likely to contain objects of interest. This pragmatic approach avoids the challenges associated with perceptual filling-in, or the completion of object shape and internal details. Rather than attempting to fill in or predict these internal details, region proposal methods assume that any given proposed region encompasses an object of interest in its entirety. The more nuanced task of identifying the exact object within the bounding box, its shape, and other characteristics, is typically delegated to subsequent processing stages in the object detection or recognition pipeline. Therefore, the mechanisms discovered in region proposal networks may be less about mirroring the brain's specific processes for figure-ground organization and more about a distinct computational approach to identifying 'objectness'. This understanding could potentially provide novel insights into the general principles of object perception, even though it might not directly map onto the mechanisms employed by the brain.


### B. Motion Processing and Terminator Integration

* 1. Neurophysiology Literature:

| Paper Title | Authors | Year | Model Organism & Visual Area | Stimulus |
|-------------|---------|------|------------------------------|----------|
| Contour integration by the human visual system: Evidence for a local “association field” | Field, Hayes, & Hess | 1993 | Human, Not specified | Various contour stimuli |
| Integration of contour and terminator signals in visual area MT of alert macaque | Pack, Livingstone, Duffy, & Born | 2003 | Macaque Monkey, Visual Area MT (V5) | Moving "barber pole" stimuli |

*  2. Possible Deep Learning Architectures:

| Paper Title | Authors | Year | Neural Network | Why relevant? |
|-------------|---------|------|----------------|----------------|
| Two-Stream Convolutional Networks for Action Recognition in Videos | Simonyan & Zisserman | 2014 | Two-Stream Convolutional Networks | Can effectively capture both spatial and temporal information |
| Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting | Shi et al. | 2015 | ConvLSTM | Combines CNNs and LSTM to process both spatial and temporal information |

* 3. Mechanism Study and Interdicipline Review:  

| Paper Title | Authors | Year | Model/Algorithm/Idea | Why relevant? |
|-------------|---------|------|---------------------|----------------|
| Neural dynamics of motion perception: Direction fields, apertures, and resonant grouping | Grossberg & Mingolla | 1993 | The role of boundary contour system (BCS) in motion perception | Provides a theoretical basis for boundaries and motion processing |
| Scale-Invariant Line Descriptors for Wide Baseline Matching | Grompone von Gioi, Jakubowicz, Morel, & Randall | 2010 | LSD: a Line Segment Detector | A computer vision algorithm for detecting line segments, which could be useful for contour integration |

* 4. Potential Challenges and Limitations: 

Processing dynamic visual stimuli like the moving "barber pole" involves the challenge of integrating spatial and temporal information. While Two-Stream Convolutional Networks and ConvLSTM can handle both types of data, they may not model the exact processes of the brain, especially considering the non-linear and complex nature of neuronal dynamics in the brain. Another potential challenge is that of matching the exact input stimulus. In the neurophysiology studies, the stimuli are very specific and might not correspond to the type of data that the deep learning models are typically trained on (e.g., natural images or videos). 


### C. High-Level Invariant Representation

* 1. Neurophysiology Literature:

| Paper Title | Authors | Year | Model Organism & Visual Area | Stimulus |
|-------------|---------|------|------------------------------|----------|
| Invariant visual representation by single neurons in the human brain | Quiroga et al. | 2005 | Human, Medial Temporal Lobe (MTL) | Images and names of celebrities, landmarks, objects, and animals |

*  2. Possible Deep Learning Architectures:

| Paper Title | Authors | Year | Neural Network | Why relevant? |
|-------------|---------|------|----------------|----------------|
| ImageNet Classification with Deep Convolutional Neural Networks | Krizhevsky et al. | 2012 | AlexNet | We can confidently expect AlexNet to be able to create an invariant representation of images across multiple layers of abstraction |

* 3. Mechanism Study and Interdicipline Review:

| Paper Title | Authors | Year | Model/Algorithm/Idea | Why relevant? |
|-------------|---------|------|---------------------|----------------|
| Building high-level features using large scale unsupervised learning | Le et al. | 2013 | Unsupervised Feature Learning | Demonstrates the ability of deep learning models to develop high-level, invariant features without supervision |

* 4. Potential Challenges and Limitations: 

CNNs and biological neural networks fundamentally differ in architecture and operation. Despite ANNs' capacity to form high-level, invariant representations of stimuli, they still operate under predefined, fixed structures. On the other hand, biological networks exhibit plasticity and constantly modify their connectivity.
