#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

###########################################################################
# SCRIPT MODIFIED BY OLIVER ALMARAZ ON JAN-2021 FOR THE PROJECT FOUND IN: #
# 			github.com/oliver-almaraz/food_container_identifier			  #
###########################################################################

import jetson.inference
import jetson.utils

import argparse
import sys


################################################
# FOR SPEECH DESCRIPTION OF IDENTIFIED OBJECTS #
################################################

# Import text-to-speech Python library
import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set speech rate (higer = faster)
engine.setProperty('rate', 100)

# OPTIONAL Set voice
#voices = engine.getProperty('voices')
#engine.setProperty('voice', voices[1].id)



# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
# output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
# font = jetson.utils.cudaFont()


################################################
# AUDIBLE INSTRUCTIONS FOR EXITING THE PROGRAM #
################################################

engine.setProperty('rate', 200)
engine.say("Welcome, for exiting this program please keep pressing for two seconds the keyboard keys 'control' and 'c'")
engine.setProperty('rate', 100)


# process frames until the user exits
while True:

	# capture the next image
	img = input.Capture()

	# classify the image
	class_id, confidence = net.Classify(img)

	# find the object description
	class_desc = net.GetClassDesc(class_id)


#######################################################
# COMMENTED OUT EVERYTHING RELATED TO A VISUAL OUTPUT #
#######################################################
	# overlay the result on the image	
	# font.OverlayText(img, img.width, img.height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
	
	# render the image
	# output.Render(img)

	# update the title bar
	# output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

	# print out performance info
	# net.PrintProfilerTimes()

###################################################
# PASS THE OBJECT'S CLASS DESCRIPTION TO PYTTSX3  #
# AND START THE VOICE SYNTH, WAIT UNTIL IT'S DONE #
###################################################

	engine.say(class_desc)
	engine.runAndWait()


	# exit on input/output EOS
	# if not input.IsStreaming() or not output.IsStreaming():
	# 	break
