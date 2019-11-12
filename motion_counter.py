#Author: Skylar Sang
#Date: 11/12/2019
#Project: IoT System for Personnel Safety

#Description:
#Background subtraction script for detecting movements without a neural network model
#Resourced from work done by pyimagesearch.com

#Example CLI command: python motion_counter.py --mode builtin --input ../../../../Videos/v3.mp4 --output output/output_02.avi

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path output video file")
ap.add_argument("-m", "--mode", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")

args = vars(ap.parse_args())


# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])
	
firstFrame = None

if args["mode"] == "builtin":
	fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
	numFrame = 0
	while(1):
		ret, frame = vs.read()
		if numFrame % 2 == 0:
			
			#Resize frame to reduce noise and improve bounding-box creation
			#Apply gaussian MOG background subtraction
			resized = imutils.resize(frame, width=600)
			frame = cv2.GaussianBlur(resized, (11, 11), 0)
			fgmask = fgbg.apply(frame)

			#Erode and Dilate image to reduce specs of noise and small particles
			fgmask = cv2.erode(fgmask, None, iterations=1)
			fgmask = cv2.dilate(fgmask, None, iterations=7)

			#Extract white areas (areas where MOG2 sees an object)
			fgmask = cv2.inRange(fgmask, 255, 255)
			cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)

			#Extract contours of blobs in filtered image, draw bounding boxes around large 
			#blobs that represent people
			for c in cnts:
				if cv2.contourArea(c) < 5000:
					continue
				(x, y, w, h) = cv2.boundingRect(c)

				cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 255), 2)

			cv2.imshow('Frame', resized)
			cv2.imshow('Sub', fgmask)
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		numFrame += 1




elif args["mode"] == "simple":
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if args["input"] is not None and frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width=500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
		gray = cv2.erode(gray, None, iterations=2)
	
		# if the first frame is None, initialize it
		if firstFrame is None:
			firstFrame = gray
			continue
		
		#GRAYSCALE MOTION DETECTING
		# compute the absolute difference between the current frame and
		# first frame
		frameDelta = cv2.absdiff(firstFrame, gray)
		frameDelta = cv2.dilate(frameDelta, None, iterations=2)

		# Working threshold for most accurate motion detection: 60
		thresh = cv2.threshold(frameDelta, 60, 255, cv2.THRESH_BINARY)[1]
	
		# Dilate the thresholded image to fill in holes, then find contours
		# on thresholded image
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
	
		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			if cv2.contourArea(c) < 2000:
				continue
	

			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			text = "Occupied"
		
		# show the output frame
		cv2.imshow("Frame", frame)
		cv2.imshow("Thresh", thresh)
		cv2.imshow("Delta", frameDelta)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

vs.release()
cv2.destroyAllWindows()