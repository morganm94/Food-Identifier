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
I captured about 100 photos for the **training**, 20 for **validation**, and just a few for **testing**, because I wanted to test the model from live camera feed.

## Re-training the model
Now that we have collected enough data, lets **re-train a pre-trained ResNet-18 model** using [**Pytorch**](https://pytorch.org/).
ResNet-18 is a [**convolutional neural network**](https://en.wikipedia.org/wiki/Convolutional_neural_network) with 18 layers. It has already been trained for image classification, so that we only have to train it with our custom data and labels:
```
$ cd jetson-inference/python/training/classification
$ python3 train.py --model-dir=models/<YOUR-MODEL> data/<YOUR-DATASET>
```
*Hint: **models/** and **data/** are relative paths, you can change them for absolute paths if you located yout data elsewere.*

The training script chooses a total of 35 epochs by default, and lasted for about 2 hours. For my model it was not enough, and I had to re-re-train the model for a total of **100 epochs**, leaving it work overnight. You can resume the training where the script left it with something like:
```
$ python3 train.py --model-dir=models/<YOUR-MODEL> data/<YOUR-DATASET> --resume /home/$USER/jetson-inference/python/training/classification/models/checkpoint.pth.tar --start-epoch 35 --epochs 100
```
*Hint: run `python3 train.py --help` for a list of arguments and options.*

## Export your model to ONNX format and test it
Once your model's training ended, it's time to test the results to see if they are precise enough. But before testing our PyTorch model with **imagenet**, we need to export it to **O**pen **N**eural **N**etwork **E**xchange format:
```
$ python3 onnx_export.py --model-dir=models/<YOUR-MODEL>
```
### Now let's test it!
```
$ imagenet.py --model=models/<YOUR-MODEL>/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/<YOUR-DATASET>/labels.txt csi://0
```
Once you are satisfied with your model's accuracy, proceed to the next step.

## Speech description using Python lib pyttsx33
### Installind dependencies
Install a **speech synthesizer** supported by [**pyttsx3**](https://pypi.org/project/pyttsx3/):
```
$ sudo apt-get install espeak
```
And, finally, the Python library for text-to-speech conversion:
```
$ sudo pip3 install pyttsx3
```
*Hint: if you are a Python developer I recommend you install this Python lib in a virtual envioment.*

### Customizing 'imagenet.py'

This repository contains a modified script of the original [imagenet.py](https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/imagenet.py) example. Basically, we need to import, initialize and barely configure the **pyttsx3** Python3 library. Since we won't need the visual feedback and we are low on system resources, I opted to comment out the code related to it.
The script is simple and generic enough for being useful as **a starting point for a lot of accesibility projects**.

## That's it!
Test your model with our new script, passing the exact same arguments you would pass to *imagenet.py*:
```
$ python3 /home/$USER/jetson-inference/python/examples/food_container_identifier.py --model=/home/$USER/jetson-inference/python/training/classification/models/food_container_identifier/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=/home/$USER/jetson-inference/python/training/classification/data/food_container_identifier/labels.txt csi://0
```
Notice how I'm now using **absolute paths** because I saved my `food_container_identifier.py` script in a different directory than my data. You can make an **alias** in `~.bashrc` or just move everything to the same directory to spare some time.
