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
Nvidia documentation is clear enough, therefore I won’t explain in detail those first steps. Instead, this video focuses on training an image classification model with our own collected data (that is food containers in our kitchen), and on using a Python library for making the Jetson Nano read out loud its guess.
