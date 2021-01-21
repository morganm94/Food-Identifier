# Food Container Identifier
Speech description of custon food containers using Nvidia's Jetson Nano.

## Introduction
Since last year, I’ve been programming some JavaScript and C to adapt braille-learning resources for Spanish-speaking students and teachers, and since I recently got my **Jetson Nano Developer Kit** I started to realize the huge potential of this little but powerful computer towards **accesibility**.
In this particular project, I will train a Neural Network to identify some food containers I have in my kitchen and also to make the Jetson Nano read out loud the name of that product.

### But why would it be useful?
Visually impaired people use other senses for understanding the world around them, that’s why they struggle to identify objects with a similar, if no identical shape. They normally deal with that keeping their food supplies, and everythin in general in a very strict order, but a friend told me that openning the wrong food can is something common and annoying. **Now have you got an idea of how frustrating it could be just to make a meal without someone around you?**
It’s true that there are several mobile aplications that can recognize most objects, but when it comes to very specific or regional objects, they fail to describe in such detail the the despcrition to be useful. For example, Google Lens will fail to decribe the can I’m holding in front of my camera if it’s not in the right angle, and it might describe just a 'can', which is usless to me. Therefore visually impaired people would probably find helpful for an autonomous life to have a device able to **indentify and describe with enough precision any of the cans and jars they normally use, no matter if it’s not in the right angle in front of the camera**. And that’s exactly what this project is about.

## First steps
For this project, you will have to follow Nvidia tutorials and documentation for [**setting up your Jetson Nano**](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit) and for [**configuring the software for training neural networks with Nvdia TensorRT optimized for the Jetson platform**](https://github.com/dusty-nv/jetson-inference).
Nvidia documentation is clear enough, therefore I won’t explain in detail those first steps. Instead, I will focus on **training an image classification model with our own collected data** (that is food containers in our kitchen), and on **using a Python library for making the Jetson Nano read out loud its guess**.
*Just a hint: If you run into memmory issues and you get some processes killed, [**build the project from source**](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md) as I did*

## Collecting our data
(This covers just the very basic procedure. For the complete documentation, visit the original [**jetson-inference repository**](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect.md)).

Select some items in your kitchen (I chose around 30), then create a new directory in `jetson-inference/python/training/classification/data` and create a new text file there named *labels.txt* with a list of your selected objects, **they must be in alphabetical order and there must be only one item (label) per line**.
Then open the camera-capture tool, select the path of your data directory and *labels.txt*, and start capturing fotos in different angles and positions, changing the background occasionally.

```
$ camera-capture csi://0       # using default MIPI CSI camera
$ camera-capture /dev/video0   # using V4L2 camera /dev/video0
```
I captured about 100 photos for the **training**, 20 for **validation**, and just a few for **testing**, because I wanted to test the model from the camera.

## Re-training the model
