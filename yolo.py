import cv2
import numpy as np

cfg_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
labels_path = 'coco.names'

LABELS = open(labels_path).read().strip().split('\n')
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

image = cv2.imread('00019.png')
(h, w) = image.shape[0:2]
processed_image = cv2.dnn.blobFromImage(image, 1./255, (416,416), swapRB=True, crop=False)

model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = model.getLayerNames()
x = model.getUnconnectedOutLayers()
output_layer = [layer_names[i[0]-1] for i in model.getUnconnectedOutLayers()]

model.setInput(processed_image)
layer_outputs = model.forward(processed_image)


bonding_boxes = []
prediction_confidence = []
classid = []
confidence_threshold = 0.3
nms_threshold = 0.4

for output in layer_outputs:
	for prediction in output:
		score = prediction[5:]
		cid = np.argmax(score)
		confidence = score[cid]
		if confidence > confidence_threshold:
			(x_center, y_center, width, height) = prediction[0:4]*np.array([w, h, w, h])
			# should be integer (or NMS gives error)
			width = int(width)
			height = int(height)
			x = int(x_center - width/2)
			y = int(y_center - height/2)
			# should be float (or NMS gives error)
			prediction_confidence.append(float(confidence))
			classid.append(cid)
			bonding_boxes.append([x, y, width, height])

indices = cv2.dnn.NMSBoxes(bonding_boxes, prediction_confidence, confidence_threshold, nms_threshold)

for ind in indices:
	ind = ind[0]
	box = bonding_boxes[ind]
	x = box[0]
	y = box[1]
	width = box[2]
	height = box[3]
	color = [int(c) for c in COLORS[classid[ind]]]
	cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
	text = "{}: {:.4f}".format(LABELS[classid[ind]], prediction_confidence[ind])
	cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow('out', image)
cv2.waitKey(0)
cv2.destroyAllWindows()