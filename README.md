# Food Container Identifier
Re-train a ResNet-18 Neural Network with PyTorch for image classification of food containers from a live camera feed and use a Python script for speech description of those food containers for the blind or visually impaired people using Nvidia's Jetson Nano.

**Note**: this project's data is too big for a GitHub repository, I had to upload it to an [external service](https://1drv.ms/u/s!AuyJyRlIYCmGhHePwGWAZX43FWU1?e=wNfQ3i). For the same reason, my model, already trained with my data and exported to ONNX format, can be found in the [releases](https://github.com/oliver-almaraz/food_container_identifier/releases/tag/v1) section.

## Introduction
Since last year, I’ve been programming some [JavaScript](https://github.com/oliver-almaraz/BrailleTermWeb) and [C](https://github.com/oliver-almaraz/Parkins) to adapt braille-learning resources for Spanish-speaking students and teachers, and since I recently got my **Jetson Nano Developer Kit** I started to realize the huge potential of this little but powerful computer towards **accessibility**.
In this particular project, I will re-train a ResNet-18 Neural Network with PyTorch to identify some food containers I have in my kitchen and also to make the Jetson Nano read out loud the name of that product.

### But why would it be useful?
Blind and visually impaired people may adapt to use other senses for understanding the world around them, that’s why they may struggle to identify objects with a similar, if not identical shape. They normally deal with that keeping their food supplies (and everything in general) in a very strict order, but a friend told me that opening the wrong food can is something common and annoying. **Now have you got an idea of how frustrating it could be just to make a meal without someone around you?**

It’s true that there are several mobile applications that can recognize most objects, but when it comes to very specific or regional objects (like most items in my kitchen), they fail to describe in such detail the description for it being useful. For example, Google Lens will fail to describe the can I’m holding in front of my camera if it’s not in the right angle, and it might describe just a 'can', which is useless to me. Therefore visually impaired people would probably find helpful for an autonomous life to have a device wich is able to **indentify and describe with enough precision any of the cans and jars they normally use, no matter if it’s not in the right angle in front of the camera**. And that’s exactly what this project is about.

## Requirements:
You will need a Jetson Nano, either the 2gb or the 4gb version. Be aware that if you have the 2gb version (like me) you might have to take [desperate measures](https://github.com/oliver-almaraz/food_container_identifier/blob/main/README.md#desperate-measures) for getting the most out of those 2gb of memory.
For the O.S. you will need a microSD card of at least 64gb and a USB-C 5v, 3A power supply. If you will access your Nano in **headless mode** you only need a microUSB cable, otherwise you will need a monitor, keyboard and mouse. Obviously, you need a computer to write the image to the microSD card and for accessing the nano via SSH, in case you will be doing so.
You will also need a **camera**. See the list of Nvidia's [officially supported cameras](https://developer.nvidia.com/embedded/jetson-partner-supported-cameras). If you get a MIPI CSI camera (like I did) you will have to get also a **camera mount** and a **tripod**. I made one myself, so if you like to drill and you have some spare wood or metal, you will have fun making one. The Rapberry Pi Camera V2 comes with a 15cm long **ribbon flex cable**, depending on your camera mount you might also need a longer one.

### My setup:

<img src="https://user-images.githubusercontent.com/69062188/105522755-3e534000-5ca3-11eb-8dce-3b0b8a7707fa.jpg" width="80%"></img> 


I'm using a Jetson Nano 2gb Developer Kit running from an SSD (thanks to [JetsonHacksNano](https://github.com/JetsonHacksNano/rootOnUSB)!), a Raspberry Pi Camera V2.1, a camera mount I made myself and a generic tripod. I also got a GeekPi 40mm, 5v, 4 pin PWM cooling fan (it's not necessary but it helped me sleep fearlessly while leaving my Jetson Nano train a model overnight). And... yes, that's a carboard box, my Jetson's case is still on its way from China.

That's it for the hardware. Now, we will be doing almost everything from the comand-line-interface (terminal), which could intimidate non-Linux users. Be not afraid, if you don't have a Linux background but you're good at following instructions you will be fine. Contact me if there's something I can make easier to follow.
We will edit a simple Python script, but even if you are not a programmer, you will be able to follow what's going on by just reading the comments in the script. Nevertheless, I encourage you to learn a bit of Python so that you can adapt this project to your own purposes.

## First steps
For this project, you will have to follow Nvidia tutorials and documentation for [**setting up your Jetson Nano**](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit) and for [**configuring the software for training neural networks with Nvdia TensorRT optimized for the Jetson platform**](https://github.com/dusty-nv/jetson-inference).
Nvidia documentation is clear enough, therefore I won’t explain in detail those first steps. Instead, I will focus on **training an image classification model with our own collected data** (that is food containers in our kitchen), and on **using a Python library for making the Jetson Nano read out loud its guess**.

*Hint: I suggest you [**build the project from source**](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md) instead of running the Docker container, while memory management is sometimes unpredictable using containers.*

## Collecting our data
(This covers just the very basic procedure. For the complete documentation, visit the original [**jetson-inference repository**](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect.md)).

Select some items in your kitchen (I chose around 30), then create a new directory in `jetson-inference/python/training/classification/data` and create a new text file there named *labels.txt* with a list of your selected objects, **they must be in alphabetical order and there must be only one item (label) per line**. (You can consult my own [*labels.txt*](https://github.com/oliver-almaraz/food_container_identifier/blob/main/labels.txt) file).
Then open the camera-capture tool, select the path of your data directory and *labels.txt*, and start capturing pictures in different angles and positions, changing the background occasionally.

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

The training script ran a total of 35 epochs by default, and lasted for about 2 hours. For my model it was not enough, and I had to re-re-train the model for a total of **100 epochs**, leaving it work overnight. You can resume the training where the script left it with something like:
```
$ python3 train.py --model-dir=models/<YOUR-MODEL> data/<YOUR-DATASET> --resume /home/$USER/jetson-inference/python/training/classification/models/checkpoint.pth.tar --start-epoch 35 --epochs 100
```
*Hint: run `python3 train.py --help` for a list of arguments and options.*

### Desperate measures
Training a model is a memory-hungry process that lasts several hours. If you're using the Jetson Nano 2gb like me, you might need to follow these next steps to prevent your process from being killed by Linux' memory management:
  1. Acess you Jetson Nano from an SSH session and stop the graphical session with:
    `$ sudo systemctl stop lightdm`
    (that will give you extra 300mb of memory)
  2. If you already have a SWAP file of at least 4gb, increase it's usage to the maximum:
    `$ sudo sysctl vm.swappiness=100`
    (keep in mind if you regularly abuse the SWAP usage it will shorten you microSD card's life)
  3. As suggested in the jetson-inference repository:
  
    to save memory, you can also reduce the --batch-size (default 8) and --workers (default 2)
    
Remember that these are **desperate measures** to follow in case your training-process gets killed.

## Export your model to ONNX format and test it
Once your model's training ended, it's time to test the results to see if they are precise enough. But before testing our PyTorch model with **imagenet**, we need to export it to **O**pen **N**eural **N**etwork **E**xchange format:
```
$ python3 onnx_export.py --model-dir=models/<YOUR-MODEL>
```
### Now let's test it!
```
$ imagenet.py --model=models/<YOUR-MODEL>/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/<YOUR-DATASET>/labels.txt csi://0
```
Once you are satisfied with your model's accuracy, proceed to the next step. (If you wish, you can also try [my own model](https://github.com/oliver-almaraz/food_container_identifier/releases/tag/v1) already trained and exported to ONNX format).

## Speech description using Python library pyttsx33
### Installing dependencies
Install a **speech synthesizer** supported by [**pyttsx3**](https://pypi.org/project/pyttsx3/):
```
$ sudo apt-get install espeak
```
Then install the **P**ackage **I**nstaller for **P**ython 3:
```
$ sudo apt-get install python3-pip
```
And, finally, the Python library for text-to-speech conversion:
```
$ sudo pip3 install pyttsx3
```
*Hint: if you are a Python developer I recommend you install this Python library in a virtual environment.*

### Customizing 'imagenet.py'

This repository contains a [modified script](https://github.com/oliver-almaraz/food_container_identifier/blob/main/food_container_identifier.py) of the original [imagenet.py](https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/imagenet.py) example.
For adding the speech description feature to our *imagenet.py* script, we need to import, initialize and configure the **pyttsx3** Python3 library (lines 40 - 51):

```python
# Import text-to-speech Python library
import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set speech rate (higer = faster)
# engine.setProperty('rate', 100)

# OPTIONAL Set voice
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)
```

After capturing and getting the class of the image from the camera, we **pass the objet's class to the voice synth engine** and wait for it to end speaking (lines 131 - 132):
```python
		engine.say(class_desc)
		engine.runAndWait()
```

Since **we won't need the visual feedback** and we are low on system resources, I opted to comment out the code related to it (lines 83 - 84):

```python
# output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
# font = jetson.utils.cudaFont()
```

And also lines 114 to 124:

```python
  # overlay the result on the image	
  # font.OverlayText(img, img.width, img.height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)

  # render the image
  # output.Render(img)

  # update the title bar
  # output.SetStatus("{:s} |   Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

  # print out performance info
  # net.PrintProfilerTimes()
```
Because we want our final program to be **accessible and easy to use**, let's add audible instructions for exiting the program with a **KeyboardInterrupt** (lines 91 - 93):

```python
engine.setProperty('rate', 200)
engine.say("Welcome, for exiting this program please keep pressing for two seconds the keyboard keys 'control' and 'c'")
engine.setProperty('rate', 100)
```
And finally, if someone is going to run our program often, it would be nice to make an **alias** for our long command. Run in a terminal:
```shell
$ echo "alias food_container_identifier='THE LONG COMMAND' #This is a comment to remember what this alias does." > ~/.bashrc
$ source ~/.bashrc
```
Now every time we run *food_container_identifier* (or any name you wrote instead) from a terminal it will be automatically replaced by the long command we wrote inside **''** and executed. I recommend you use absolute paths in this case.
Keyboard bindings are important to have in mind when developing accessiible programs, you might also want to set one for opening a terminal (normally, it defaults to **Ctrl + Alt + t**).

The script is simple and generic enough for being useful as **a starting point for a lot of accessibility projects**.
Please take a look at the [Python script](https://github.com/oliver-almaraz/food_container_identifier/blob/main/food_container_identifier.py) even if you are not a programmer, and try to understand what's going on.


## That's it!
Test your model with our new script, passing the exact same arguments you would pass to *imagenet.py*:
```
$ python3 /home/$USER/jetson-inference/python/examples/food_container_identifier.py --model=/home/$USER/jetson-inference/python/training/classification/models/food_container_identifier/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=/home/$USER/jetson-inference/python/training/classification/data/food_container_identifier/labels.txt csi://0
```
Notice how I'm now using **absolute paths** because I saved my `food_container_identifier.py` script in a different directory than my data. You can make an **alias** in `~.bashrc` or just move everything to the same directory to spare some time.

## About audio output
The Jetson Nano does not have a 3.5mm audio port nor a Bluetooth module, therefore we will have to use non-conventional audio outputs. If you have an HDMI monitor you can use its speakers. Because I only have a VGA monitor, I needed an HDMI to VGA converter, which connects to the Jetson Nano through the HDMI port and has two outputs: a VGA interface and a 3.5mm audio port, to which you can connect a speaker or headphones.
You can also get a cheap external sound card, which connects to a Jetson's USB port and has a 3.5mm audio output (image below).

<img src="https://user-images.githubusercontent.com/69062188/105522763-40b59a00-5ca3-11eb-8520-ea1088ce004b.jpg" width="40%"></img> 

No matter which audio output device you use, go to the **mixer** (right click the speaker's icon on the bottom-right corner and then *launch Mixer*, or run ´pavucontrol´ from the terminal) and select the right output device for **ALSA plug-in [aplay]** under the *Playback* tab **while our program is running** (only the currently playing programs are shown in the mixer).
