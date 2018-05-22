#!/usr/bin/env python


#Importing modules
import numpy as np
import cv2
import time

# local modules
from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''



#=======================================================================
#Different function for detection
#======================================================================

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=(210, 210), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#-------------

def detect1(img, cascade):
    global E1
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    E1 = 1
    return rects

#-------------

def detect2(img, cascade):
    global E2
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    E2 = 1
    return rects

#-------------------------------
#Function for drawing rectangle
#-------------------------------
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#=======================================================================

if __name__ == '__main__':
    import sys, getopt
    print help_message

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])

#-----------------------------------
#Selecting video source(file/camera)
#-----------------------------------

    try:
        video_src = video_src[0]
    except:


    	#fn = raw_input("Enter Camera/Video (1/0): ")
	fn = 0#int(fn)
	if fn == 1:
		print "Video File "
		cam = create_capture('vania.mp4', fallback='synth:bg=../data/lena.jpg:noise=0.05')

	elif fn == 0:
		print "Web Camera"
		cam = create_capture(fn, fallback='synth:bg=../data/lena.jpg:noise=0.05')

#------------------------
#Defining various cascade
#------------------------

    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt_tree.xml")
    nested_fn  = args.get('--nested-cascade', "cascade/open.xml")
    nested1_fn  = args.get('--nested-cascade', "cascade/opencnlosed.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    nested1 = cv2.CascadeClassifier(nested1_fn)

#------------------------------
#Variables for counting & reset
#------------------------------

    xx1 = 0
    yy1 = 0
    xx2 = 200
    yy2 = 200
    b1 = 0

    begin=time.time()
    x10 = 0
    x11 = 0
    x20 = 0
    x21 = 0
    timeout = 1

#=======================================================================
#starting while loop
#-------------------
    while True:
	global E1
        ret, img = cam.read() #Readin image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting to gray scale
        gray = cv2.equalizeHist(gray)
	x30 = 0
	E1 = 0
	E2 = 0
        t = clock()
        rects = detect(gray, cascade) #activate function to detect face
        vis = img.copy() #copying image
	vistest = img.copy()
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2] #croping image
                vis_roi = vis[y1:y2, x1:x2]
		vistest_roi = vistest[y1:y2, x1:x2]

		#cv2.imshow('face', vis_roi)
                subrects = detect1(roi.copy(), nested) #detecting closed eye
                subrects1 = detect2(roi.copy(), nested1) #detecting closed and open eye

		draw_rects(vistest_roi, subrects1, (255, 0, 0))
		b1 = 1
                draw_rects(vis_roi, subrects, (255, 0, 0)) #Draw blue rectangle in eye region
		dt = clock() - t #Time difference

		if b1==1:
			draw_str(vis, (20, 40), 'Count: %d' %(x11)) #Write count string to output	
			x30 = 1
			if E1==1 and E2==0: #Eye Closed
				draw_rects(vis, rects, (0, 0, 255)) #Draw red rectangle
				draw_str(vis, (20, 20), 'CLOSED')	
				x10 = x10+1 #variable to count eye blink
				if x20==1: # To start timer when detect a closed eye
					x20 = 0
					begin=time.time()
				if time.time()-begin > timeout: #Time out to start beep
					print 'beep'
					x21 = 1
					draw_str(vis, (120, 20), 'SLEEPING')	



			if E1==1 and E2==1: #Eye Open
				draw_rects(vis, rects, (0, 255, 0)) #Draw Green rectangle
				draw_str(vis, (20, 20), 'OPEN')	
				x20 = 1
				if x10>1: #Count eye blink
					x11 = x11+1
					x10 = 0					
				if x21==1: #used to turn of beep when detecting an open eye
					print 'beep off'
					x21 = 0
							

        if x30==0 and x21==1: # x20 and x21 used to turn off beep when face suddenly dissapear while beep is on
		print 'beep off'
		x21 = 0
		x20 = 1


	


        cv2.imshow('facedetect', vis) #show the result
	#cv2.imshow('face1', vistest)

        if 0xFF & cv2.waitKey(5) == 27: #Esc key to break
            break

#=======================================================================
    cv2.destroyAllWindows()
