import numpy as np
import random
import os
import csv
import sys


def box_intersection1(a, b):
	area = 1
	if (b[0] >= a[0] and b[0] <= a[2]): # (b[3] >= a[3] and b[3] <= a[5]):
		xmin = b[0]
		xmax = min(b[2],a[2])
	elif (a[0] >= b[0] and a[0] <= b[2]):
		xmin = a[0]
		xmax = min(b[2],a[2])
	else:
		area = 0
	if (b[1] >= a[1] and b[1] <= a[3]):
		ymin = b[1]
		ymax = min(a[3],b[3])
	elif (a[1] >= b[1] and a[1] <= b[3]):
		ymin = a[1] 
		ymax = min(a[3],b[3])
	else:
		area = 0
	if (area > 0):
		area = (xmax - xmin)*(ymax - ymin)
	return area

def box_union(a, b):
	area_a = (a[2] - a[0])*(a[3] - a[1])
	area_b = (b[2] - b[0])*(b[3] - b[1])
	union_ab = area_a + area_b - box_intersection1(a,b)
	return union_ab

def compute_iou(a, b):
	iou = box_intersection1(a,b)/box_union(a,b)
	return iou

def extract_boxes():
	boundingboxes = list()
	csv_fname = os.path.join('/Users/Kush/Desktop/PyTorch-Python/udacity/darkflow/udacity_train_2.csv')
	with open(csv_fname, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )

		for row in spamreader:
			labels = row[1:]

			for i in range(0, len(labels), 5):
				boundbox = list()
				xmin = int(labels[i])
				ymin = int(labels[i + 1])
				xmax = int(labels[i + 2])
				ymax = int(labels[i + 3])
				boundbox = [xmin, ymin, xmax, ymax]
				boundingboxes.append(boundbox)
	return boundingboxes
            
def compute_distances(boxes, centroids):
	dist = np.ndarray(shape = (len(boxes),len(centroids)), dtype = float)
	for i in range(len(boxes)):
		for j in range(len(centroids)):
			dist[i,j] = 1 - compute_iou(boxes[i], centroids[j])
	return dist

def compute_centroids_from_assignment(assignments, boxes, num_classes):
	boxes_arr = np.asarray(boxes)
	centroids = np.zeros((5,4),dtype=float)
	#print('boxes_arr size: ', boxes_arr.shape)
	for i in range(num_classes):
		centroids[i,:] = np.mean(boxes_arr[assignments == i, :], axis = 0)
	return centroids

def compute_w_h(centroids, num_classes, s, input_size, W, H):
	ratio_x = float(W/input_size)
	ratio_y = float(H/input_size)
	cents = np.zeros((num_classes,2),dtype = float)
	for i in range(num_classes):
		cents[i,0] = (centroids[i][2] - centroids[i][0])/(s * ratio_x)
		cents[i,1] = (centroids[i][3] - centroids[i][1])/(s * ratio_y) 
	return cents

def compute_centroids(num_classes):
	W = 1920
	H = 1200
	S = 13
	input_size = 416
	boundingboxes = extract_boxes();
	init_indices = random.sample(range(0,len(boundingboxes)),num_classes)
	cents = [boundingboxes[i] for i in init_indices]
	centroids = np.asarray(cents)
	#print ('centroids size: ', centroids.shape)
	distances = np.ndarray(shape = (len(boundingboxes),num_classes),dtype = float)
	assignment = np.zeros((len(boundingboxes),1), dtype = int)
	prev_assignment = np.ones((len(boundingboxes),1))
	distances = compute_distances(boundingboxes, centroids)
	iterations = 70
	for i in range(iterations):
		assignment = np.argmin(distances, axis = 1)
		centroids = compute_centroids_from_assignment(assignment, boundingboxes, num_classes)
		distances = compute_distances(boundingboxes, centroids)
		print ('iteration: ', i)
		if (((i+1) % 10) == 0):
			print ('centroid: ')
			print (compute_w_h(centroids,num_classes,S,input_size,W,H))
	cents = compute_w_h(centroids,num_classes,S,input_size,W,H)
	print (cents)
	print (centroids)
	return cents

cent = compute_centroids(5)
print (cent)
   


