import numpy as np
import math
import cv2
import os
#from scipy.special import expit
from utils.box import BoundBox, box_iou, prob_compare
from utils.box import prob_compare2, box_intersection
import json

	
_thresh = dict({
	'person': .2,
	'pottedplant': .1,
	'chair': .12,
	'tvmonitor': .13
})

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	# meta
	print ('yolo 2 postprocess')
	print ('img', im)
	meta = self.meta
	H, W, _ = meta['out_size']
	threshold = meta['thresh']
	C, B = meta['classes'], meta['num']
	anchors = meta['anchors']
	net_out = net_out.reshape([H, W, B, -1])

	boxes = list()
	for row in range(H):
		for col in range(W):
			for b in range(B):
				bx = BoundBox(C)
				bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
				bx.c = expit(bx.c)
				bx.x = (col + expit(bx.x)) / W
				bx.y = (row + expit(bx.y)) / H
				bx.w = math.exp(bx.w) * anchors[2 * b + 0] / W
				bx.h = math.exp(bx.h) * anchors[2 * b + 1] / H
				classes = net_out[row, col, b, 5:]
				#print('classes',classes)
				bx.probs = _softmax(classes) * bx.c
				#print ('bx.probs: ', bx.probs)
				bx.probs *= bx.probs > threshold
				#print ('bx.probs after threshold: ', bx.probs)
				boxes.append(bx)

	# non max suppress boxes
	for c in range(C):
		for i in range(len(boxes)):
			boxes[i].class_num = c
		boxes = sorted(boxes, key = prob_compare)
		for i in range(len(boxes)):
			boxi = boxes[i]
			if boxi.probs[c] == 0: continue
			for j in range(i + 1, len(boxes)):
				boxj = boxes[j]
				if box_iou(boxi, boxj) >= .4:
					boxes[j].probs[c] = 0.


	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	for b in boxes:
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = 'object' * int(C < 2)
		label += labels[max_indx] * int(C>1)
		if max_prob > threshold:
			left  = int ((b.x - b.w/2.) * w)
			right = int ((b.x + b.w/2.) * w)
			top   = int ((b.y - b.h/2.) * h)
			bot   = int ((b.y + b.h/2.) * h)
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1
			thick = int((h+w)/300) 
			#print ('drawing rectangle')
			cv2.rectangle(imgcv, 
				(left, top), (right, bot), 
				colors[max_indx], thick)
			mess = '{}'.format(label)
			cv2.putText(imgcv, mess, (left, top - 12), 
				0, 1e-3 * h, colors[max_indx],thick//3)


	if not save: return imgcv
	outfolder = os.path.join(self.FLAGS.test, 'out') 
	img_name = os.path.join(outfolder, im.split('/')[-1])
	cv2.imwrite(img_name, imgcv)



def postprocess(self, net_out, im, ground_truth, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
  
	predictboxes = list()  # for storing predicted positive results

	meta = self.meta
	H, W, _ = meta['out_size']
	threshold = meta['thresh']
	C, B = meta['classes'], meta['num']
	anchors = meta['anchors']
	net_out = net_out.reshape([H, W, B, -1])
	boxes = list()
	for row in range(H):
		for col in range(W):
			for b in range(B):
				bx = BoundBox(C)
				bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
				bx.c = expit(bx.c)
				bx.x = (col + expit(bx.x)) / W
				bx.y = (row + expit(bx.y)) / H
				bx.w = math.exp(bx.w) * anchors[2 * b + 0] / W
				bx.h = math.exp(bx.h) * anchors[2 * b + 1] / H
				classes = net_out[row, col, b, 5:]
				bx.probs = _softmax(classes) * bx.c
				bx.probs *= bx.probs > threshold
				boxes.append(bx)

	# non max suppress boxes
	for c in range(C):
		for i in range(len(boxes)):
			boxes[i].class_num = c
		boxes = sorted(boxes, key = prob_compare)
		for i in range(len(boxes)):
			boxi = boxes[i]
			if boxi.probs[c] == 0: continue
			for j in range(i + 1, len(boxes)):
				boxj = boxes[j]
				if box_iou(boxi, boxj) >= .4:
					boxes[j].probs[c] = 0.


	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	resultsForJSON = []
	
	num_classes = 5
	sum_iou = 0
	total_boxes = 0
	sum_iou_cl = np.zeros((num_classes,1), dtype = float)
	total_boxes_cl = np.zeros((num_classes,1), dtype = int)
	
	for b in boxes:
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = 'object' * int(C < 2)
		label += labels[max_indx] * int(C>1)
		if max_prob > threshold:
			left  = int ((b.x - b.w/2.) * w)
			right = int ((b.x + b.w/2.) * w)
			top   = int ((b.y - b.h/2.) * h)
			bot   = int ((b.y + b.h/2.) * h)
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1
			thick = int((h+w)/300)
			predict_box = [label, max_indx, left, top, right, bot]
			iou = compute_iou_from_boxes(predict_box, ground_truth)
			sum_iou += iou
			sum_iou_cl[max_indx] += iou
			total_boxes_cl[max_indx] += 1
			total_boxes += 1
			# storing bounding box info in json output 
			resultsForJSON.append({"label": label, "confidence": float('%.2f' % iou), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			predictboxes.append(predict_box)
			cv2.rectangle(imgcv, 
				(left, top), (right, bot), 
				colors[max_indx], thick)
			mess = '{}'.format(label)
			cv2.putText(imgcv, mess, (left, top - 12), 
				0, 1e-3 * h, colors[max_indx],thick//3)


    # writing json output to a file
	outfolder = os.path.join(self.FLAGS.test, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	textJSON = json.dumps(resultsForJSON)
	textFile = os.path.splitext(img_name)[0] + ".json"
	with open(textFile, 'w') as f:
		f.write(textJSON)

	thick = int((h+w)/300)

	num_classes = 5
	iou_thresh = 0.40
	true_positives, false_positives, true_positives_cl, false_positives_cl, total_positives_cl = compute_true_and_false_positives(predictboxes, ground_truth, num_classes, iou_thresh)

	false_negatives, false_negatives_cl = compute_false_negatives(predictboxes, ground_truth, num_classes, iou_thresh)

	if not save: return imgcv
	outfolder = os.path.join(self.FLAGS.test, 'out') 
	img_name = os.path.join(outfolder, im.split('/')[-1])
	cv2.imwrite(img_name, imgcv)
	return true_positives, false_positives, false_negatives, true_positives_cl, false_positives_cl, total_positives_cl, false_negatives_cl, sum_iou, sum_iou_cl, total_boxes, total_boxes_cl


# this function computes true and false positives given predicted boxes and ground truth boxes

def compute_true_and_false_positives(predictboxes, ground_truth, num_classes, iou_thresh):
	true_positives = 0
	false_positives = 0
	true_positives_cl = np.zeros((num_classes,1),dtype = int)
	total_positives_cl = np.zeros((num_classes,1),dtype = int)
	false_positives_cl = np.zeros((num_classes,1),dtype = int)
	for i in range(len(predictboxes)):
		class_detect = predictboxes[i][1]
		true_positive = 0
		ratio_max = 0
		total_positives_cl[class_detect] += 1
		for j in range(len(ground_truth)):
			if predictboxes[i][0] == ground_truth[j][0]:
				ratio = compute_iou(predictboxes[i], ground_truth[j])
				if ratio >= iou_thresh:
					true_positive += 1
					if ratio > ratio_max:
						ratio_max = ratio
		if true_positive > 0:
			true_positives += 1
			true_positives_cl[class_detect] += 1
		else:
			false_positives += 1
	false_positives_cl = total_positives_cl - true_positives_cl
	return true_positives, false_positives, true_positives_cl, false_positives_cl, total_positives_cl


# this finction computes false negatives for the whole image and for the respective classes as well

def compute_false_negatives(predictboxes, ground_truth, num_classes, iou_thresh):
	false_negatives = 0
	false_negatives_cl = np.zeros((num_classes,1),dtype = int)
	for i in range(len(ground_truth)):
		class_detect = ground_truth[i][1]
		true_positive = 0
		for j in range(len(predictboxes)):
			if ground_truth[i][0] == predictboxes[j][0]:
				ratio = compute_iou(predictboxes[j], ground_truth[i])
				if ratio >= iou_thresh:
					true_positive = 1
		if true_positive == 0:
			false_negatives += 1
			false_negatives_cl[class_detect] += 1
	return false_negatives, false_negatives_cl


# computes intersection area given bounding box info
def box_intersection1(a, b):
	area = 1
	if (b[2] >= a[2] and b[2] <= a[4]): # (b[3] >= a[3] and b[3] <= a[5]):
		xmin = b[2]
		xmax = min(b[4],a[4])
	elif (a[2] >= b[2] and a[2] <= b[4]):
		xmin = a[2]
		xmax = min(b[4],a[4])
	else:
		area = 0
	if (b[3] >= a[3] and b[3] <= a[5]):
		ymin = b[3]
		ymax = min(a[5],b[5])
	elif (a[3] >= b[3] and a[3] <= b[5]):
		ymin = a[3] 
		ymax = min(a[5],b[5])
	else:
		area = 0
	if (area > 0):
		area = (xmax - xmin)*(ymax - ymin)
	return area

# computes the union area of two boxes
def box_union(a, b):
	area_a = (a[4] - a[2])*(a[5] - a[3])
	area_b = (b[4] - b[2])*(b[5] - b[3])
	union_ab = area_a + area_b - box_intersection1(a,b)
	return union_ab

# computes iou
def compute_iou(a, b):
	iou = box_intersection1(a,b)/box_union(a,b)
	return iou

# computes iou by comparing to all ground truth boxes
def compute_iou_from_boxes(a,ground_truth_boxes):
	iou = 0
	for i in range(len(ground_truth_boxes)):
		iou = max(iou,compute_iou(a,ground_truth_boxes[i]))
	return iou

def compute_overlap_ratio(a,b):
	area_a = (a[4] - a[2])*(a[5] - a[3])
	area_b = (b[4] - b[2])*(b[5] - b[3])
	ratio = box_intersection1(a,b)/(min(area_a,area_b) + 0.001)
	return ratio