import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# apply brightness/contrast setting to an image
# range between 0 and 127
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
	if brightness != 0:
		if brightness > 0:
			shadow = brightness
			highlight = 255
		else:
			shadow = 0
			highlight = 255 + brightness
		alpha_b = (highlight - shadow)/255
		gamma_b = shadow
		
		buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
	else:
		buf = input_img.copy()

	if contrast != 0:
		f = 131*(contrast + 127)/(127*(131-contrast))
		alpha_c = f
		gamma_c = 127*(1-f)

		buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

	return buf

# now we invoke the well-known Benson algorithm, to determine the sequence of the rectangles
# usage: bestRect = generate_tot_points(boundRectClosest), for instance
eps = 10 # error margin
small = 1e-6
def generate_tot_points(unordered_boxes):
	num_char = 0 # denotes number of characters 
	line_type = 0 # number of lines

	x = np.zeros(10)
	y = np.zeros(10)
	boxes = []
	num_box = 0
	for box in unordered_boxes: # for each bounding box
		#print(box)
		x[num_box] = box[0]
		y[num_box] = box[1]
		boxes.append(box)
		num_box += 1
	#print(x)
	#print(y)
	#test_box = []
	#for i in range(0,num_box):
	#	test_box.append((x[i],y[i],i))
	#print_box(img,test_box,boxes)

	# line is in the form ax+by+c=0
	final_a = 0
	final_b = 0
	final_c = 0
	final_points = []

	# first, check if 1 line can fit 7 / 8 boxes
	for i in range(0,num_box):
		for j in range(i+1,num_box):
			x1 = x[i]
			y1 = y[i]
			x2 = x[j]
			y2 = y[j]

			a = (y[i]-y[j])
			b = (x[j]-x[i])
			c = x[i]*y[j] - x[j]*y[i]
			check = 0
			line_points = []

			for k in range(0,num_box):
				diff = (a*x[k] + c)/(b+small)+y[k]
				if (abs(diff) <= eps):
					check += 1
					line_points.append((x[k],y[k],boxes[k][2],boxes[k][3]))

			if check >= 7 and len(final_points) < len(line_points): # line misses at most 1 point
				final_a = a
				final_b = b
				final_c = c
				final_points = line_points


	if final_a != 0 or final_b != 0 or final_c != 0:
		line_type = 1
	else:
		line_type = 2

	if line_type == 1:

		final_points.sort()
		return final_points
		#print(sort_box)
	else:
		# from here on, we assume that the first line has 3 characters
		# first line
		line1_a = 0
		line1_b = 0
		line1_c = 0
		line1_points = []

		# second line
		line2_a = 0
		line2_b = 0
		line2_c = 0
		line2_points = []
		num_line2 = 0

		for i in range(0,num_box):
			for j in range(i+1,num_box):
				x1 = x[i]
				y1 = y[i]
				x2 = x[j]
				y2 = y[j]

				a = (y[i]-y[j])
				b = (x[j]-x[i])
				c = x[i]*y[j] - x[j]*y[i]
				check = 0

				line_points = []
				for k in range(0,num_box):
					diff = (a*x[k] + c)/(b+small)+y[k]
					if (abs(diff) <= eps): # count number that are close to the line
						check += 1
						line_points.append((x[k],y[k],boxes[k][2],boxes[k][3]))

				if check == 3: # this is the first line
					line1_a = a
					line1_b = b
					line1_c = c
					line1_points = line_points

				elif check >= 4 and check <= 5 and len(line2_points) <= len(line_points): # this is the second line
					line2_a = a
					line2_b = b
					line2_c = c
					line2_points = line_points

		line1_points.sort()
		line2_points.sort()
		tot_points = line1_points+line2_points
		return tot_points

# for a given brightness/contrast setting, run detection up till the sequence detection step
def identify_brightness_contrast(img,brightness=75,contrast=100):

	# output plot
	#fig,ax = plt.subplots(3,3)

	#ax[0][0].set_title('Original cropped image')
	#ax[0][0].imshow(img)

	# increase image contrast
	# this is necessary for the binary thresholding to work decently well
	# this is based on a few images I tested
	img_adjusted = apply_brightness_contrast(img, brightness, contrast)
	#ax[0][1].set_title('Brightness/contrast adjustment')
	#ax[0][1].imshow(img_adjusted)

	# change to grayscale
	img_adjusted = cv.cvtColor(img_adjusted, cv.COLOR_BGR2GRAY)
	#ax[0][2].set_title('Grayscaled image')
	#ax[0][2].imshow(img_adjusted)

	# perform image thresholding
	# convert image to binary
	ret, img_binary = cv.threshold(img_adjusted,127,255,cv.THRESH_BINARY)
	#ax[1][0].set_title('Binary thresholding')
	#ax[1][0].imshow(img_binary)
	
	# find contours on the image
	contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
	
	# approximate contours to rectangles
	# each of these rectangles bound a letter
	contours_poly = [None]*len(contours)
	boundRect = [None]*len(contours)
	for i, c in enumerate(contours):
		contours_poly[i] = cv.approxPolyDP(c, 3, True)
		boundRect[i] = cv.boundingRect(contours_poly[i]) # returns (x, y, w, h) - x and y are for the top left corner of the rect
	
	# draw out the bounding rectangles
	#ax[1][1].set_title('Contour detection')
	#ax[1][1].imshow(img_binary)
	#for i in range(len(boundRect)):
	#	rect = patches.Rectangle((boundRect[i][0],boundRect[i][1]),boundRect[i][2],boundRect[i][3],linewidth=1,edgecolor='r',facecolor='none')
	#	ax[1][1].add_patch(rect)


	# boundRect is a list of (x, y, w, h) for each rectangle - x and y are for the top left corner of the rect

	# issue: it also detects the parts of the car (outside the boundaries of license plate)
	# correct this by choosing the rectangles with similar areas
	# 8 best rectangles (assume license plate has 8 characters)

	total_area = np.size(img, 0) * np.size(img, 1)
	# condition for the area of the rectangle to be more than 1/150 of the total area (so we don't pick useless stuff)
	boundRectFiltered = [x for x in boundRect if x[2]*x[3]>total_area/150]

	# to try to figure out which 8 rectangles are the actual license plate characters,
	# we can try to pick out the best 12 rects based on height first
	# then from there we pick out the best 8 rects based on area
	heights = [x[3] for x in boundRectFiltered]
	midHeight = np.median(heights)
	heightsDiff = np.array([abs(x - midHeight) for x in heights])
	closestHeights = heightsDiff.argsort()[:10]
	boundRectFiltered2 = [boundRectFiltered[x] for x in closestHeights]

	areas = [x[2]*x[3] for x in boundRectFiltered2]
	midArea = np.median(areas) # median area
	areasDiff = np.array([abs(x - midArea) for x in areas])
	closestAreas = areasDiff.argsort()[:8] # indices of 8 rectangles whose areas have smallest deviation from median
	boundRectClosest = [boundRectFiltered2[x] for x in closestAreas]


	#ax[1][2].set_title('Select 8 best rects')
	#ax[1][2].imshow(img_binary)
	#boundRectShow = boundRectClosest;
	#for i in range(len(boundRectShow)):
	#	rect = patches.Rectangle(
	#		(boundRectShow[i][0],boundRectShow[i][1]),
	#		boundRectShow[i][2],boundRectShow[i][3],
	#		linewidth=1,edgecolor='red',facecolor='none')
	#	ax[1][2].add_patch(rect)

	# draw out the sequence of bounding rectangles
	# display: as we progress, the rectangle gets whiter and thinner
	#ax[2][0].set_title('Sequence detection')
	#ax[2][0].imshow(img_binary)
	#outArr = []
	#for i in range(len(bestRect)):
	#	rect = patches.Rectangle(
	#		 (bestRect[i][0],bestRect[i][1]),
	#		 bestRect[i][2],bestRect[i][3],linewidth=0.2+1.4*(8-i),edgecolor=str(0.2+i*0.1),facecolor='none')
	#	ax[2][0].add_patch(rect)
	#	outArr.append([
	#		 [bestRect[i][0],bestRect[i][1]],
	#		 [bestRect[i][0]+bestRect[i][2],bestRect[i][1]+bestRect[i][3]]
	#	 ]);

	#plt.show();
	return boundRectClosest;

def identify(img):
	blist=[-75,-50,-25,0,0,0,25,50,75] #list of brightness values
	clist=[100,75,50,0,50,100,50,75,100] #list of contrast values
	detectedRects=[] #(x,y,w,h)
	for i in range(len(blist)):
		detectedRects.append(identify_brightness_contrast(img,blist[i],clist[i]))
	
	# now choose optimum detected rects based on
	# 1. number of detected rects (filter to ensure 8)
	detectedRectsFiltered = [rects for rects in detectedRects if len(rects) >= 5 and len(rects) <= 8]
	if len(detectedRectsFiltered)==0:
		detectedRectsFiltered = detectedRects;
	
	# 2. select smallest standard deviation in height
	detectedRectsStdev = [np.std([rect[3] for rect in rects]) for rects in detectedRectsFiltered]
	smallestStdevIndex = detectedRectsStdev.index(min(detectedRectsStdev))
	
	bestRect = generate_tot_points(detectedRectsFiltered[smallestStdevIndex])
	outArr = []
	
	fig,ax = plt.subplots(3,3)
	ax[2][0].set_title('Sequence detection')
	ax[2][0].imshow(img)
	
	for i in range(len(bestRect)):
		rect = patches.Rectangle(
			(bestRect[i][0],bestRect[i][1]),
			bestRect[i][2],bestRect[i][3],linewidth=0.2+1.4*(8-i),edgecolor=str(0.2+i*0.1),facecolor='none')
		ax[2][0].add_patch(rect)
		outArr.append([
			 [bestRect[i][0],bestRect[i][1]],
			 [bestRect[i][0]+bestRect[i][2],bestRect[i][1]+bestRect[i][3]]
		 ]);

	#plt.show();
	return outArr
	