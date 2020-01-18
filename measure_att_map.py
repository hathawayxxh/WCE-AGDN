"""
Using three metrics to evaluate the attention maps.
1. precision-recall curves.
2. F-measure.
3. Mean absolute error.
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def prec_rec(att_path, gt_path, num_samples, mode):
	"""
	Binarize the saliency map with a threshold sliding from 0 to 255
	and then compare the binary maps with the ground truth.
	"""
	precision = np.zeros(255)
	recall = np.zeros(255)

	for img in range(num_samples):
		img += 200
		att_map = cv2.imread(os.path.join(att_path, str(img) + '_' + mode + '.jpg'), 0)
		att_map = 255.0 * (att_map - np.min(att_map))/(np.max(att_map) - np.min(att_map))

		gt_map = cv2.imread(os.path.join(gt_path, str(img) + '.jpg'), 0)
		gt_map = np.where(gt_map == 0, np.zeros_like(gt_map), np.ones_like(gt_map))

		for iter in range(255):
			pred = np.where(att_map >= iter, np.ones_like(att_map), np.zeros_like(att_map))

			prec = float(np.sum(pred * gt_map))/np.sum(pred + 1e-10)
			rec = float(np.sum(pred * gt_map))/np.sum(gt_map + 1e-10)

			precision[iter] += prec
			recall[iter] += rec

	precision = precision/num_samples
	recall = recall/num_samples

	return recall, precision


def plot_curves(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, title, xlabel, ylabel): #

	plt.plot(x1, y1, color = "green", label = "avg4")
	plt.plot(x2, y2, color = "green", linestyle = "--", label = "avg3")

	# plt.plot(x3, y3, color = "orange", label = "max4")
	# plt.plot(x4, y4, color = "orange", linestyle = "--", label = "max3")
	# plt.plot(x5, y5, color = "red", label = "abs4")
	# plt.plot(x6, y6, color = "red", linestyle = "--", label = "abs3")

	# plt.plot(x3, y3, color = "blue", label = "s_tri4")
	# plt.plot(x4, y4, color = "blue", linestyle = "--", label = "s_tri3")

	# plt.plot(x5, y5, color = "red", label = "c_tri4")
	# plt.plot(x6, y6, color = "red", linestyle = "--", label = "c_tri3")

	plt.plot(x5, y5, color = "red", label = "mix_rescale_s")
	plt.plot(x6, y6, color = "red", linestyle = "--", label = "mix_rescale_c")	

	plt.plot(x7, y7, color = "orange", label = "mix_s")
	plt.plot(x8, y8, color = "orange", linestyle = "--", label = "mix_c")

	# plt.plot(x9, y9, color = "black", label = "tri_mix")	

	plt.legend()
	plt.title(title)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim((0.0, 1.0))
	plt.ylim((0.0, 1.0))



def plot_pr_curve(att_path, gt_path, num_samples):

	recall1, precision1 = prec_rec(att_path + "saliency_maps", gt_path, num_samples, mode = "avg_net1")
	# recall2, precision2 = prec_rec(att_path + "saliency_maps", gt_path, num_samples, mode = "avg_net1")

	plt.plot(recall1, precision1, color = "green", label = "split1_2DTOA_net1")
	# plt.plot(recall2, precision2, color = "red", label = "split2_3DTOA_net1")

	plt.legend()
	# plt.title("Bleeding_200images")

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim((0.0, 1.0))
	plt.ylim((0.0, 1.0))

	plt.show()


def F_measure(att_path, gt_path, num_samples, mode):
	precision = 0.0
	recall = 0.0

	for img in range(num_samples):
		img += 200
		att_map = cv2.imread(os.path.join(att_path, str(img) + '_' + mode + '.jpg'), 0)
		att_map = 255.0 * (att_map - np.min(att_map))/(np.max(att_map) - np.min(att_map))
		threshold = 2 * np.mean(att_map)
		# print threshold
		pred = np.where(att_map >= threshold, np.ones_like(att_map), np.zeros_like(att_map))

		gt_map = cv2.imread(os.path.join(gt_path, str(img) + '.jpg'), 0)
		gt_map = np.where(gt_map == 0, np.zeros_like(gt_map), np.ones_like(gt_map))

		cv2.imwrite("./temporary/" + str(img) + "_" + mode + "_att.jpg", pred * 255)
		cv2.imwrite("./temporary/" + str(img) + "_gt.jpg", gt_map * 255)

		precision += float(np.sum(pred * gt_map))/np.sum(pred + 1e-10)
		recall += float(np.sum(pred * gt_map))/np.sum(gt_map + 1e-10)

	precision = precision/num_samples
	recall = recall/num_samples

		# precision = float(np.sum(pred * gt_map))/np.sum(pred + 1e-10)
		# recall = float(np.sum(pred * gt_map))/np.sum(gt_map + 1e-10)
	beta_2 = 1.0
	F_beta = (1 + beta_2) * precision * recall/(beta_2 * precision + recall + 1e-10)

	print (mode, "Precision:", precision, "Recall:", recall, "F_measure:", F_beta)




def compare_F_measure(att_path, gt_path, num_samples):

	print ("Attention F1_measure")
	# F_measure(att_path + "split1_B2_saliency", gt_path, num_samples, mode = "avg_net1")
	F_measure(att_path + "saliency_maps", gt_path, num_samples, mode = "avg_net1")


def MAE_measure(att_path, gt_path, num_samples, mode):
	"""
	Mean absolute error.
	Average difference between the prediction and ground truth in pixel level.
	"""
	MAE_error = 0.0
	for img in range(num_samples):
		img += 200
		att_map = cv2.imread(os.path.join(att_path, str(img) + '_' + mode + '.jpg'), 0)
		pred = (att_map - np.min(att_map))/(np.max(att_map) - np.min(att_map)).astype(float)

		gt_map = cv2.imread(os.path.join(gt_path, str(img) + '.jpg'), 0)
		gt_map = np.where(gt_map == 0, np.zeros_like(gt_map), np.ones_like(gt_map)).astype(float)

		MAE_error += np.mean(np.abs(pred - gt_map))

	MAE_error = MAE_error/num_samples
	print (mode, "MAE_error:", MAE_error)



def compare_MAE(att_path, gt_path, num_samples):

	print ("Mean absolute error:")
	# MAE_measure(att_path + "split1_B2_saliency", gt_path, num_samples, mode = "avg_net1")
	MAE_measure(att_path + "saliency_maps", gt_path, num_samples, mode = "avg_net1")


#####################################
### Run functions
#####################################
plot_pr_curve(att_path = "/home/xxh/Documents/Journal/code/2densenet/visualize_conf_maps/", 
	gt_path = "/home/xxh/Documents/Journal/dataset/sum_train_test_only/test_gt/test4_gt_128_dilate/", num_samples = 400)

compare_F_measure(att_path = "/home/xxh/Documents/Journal/code/2densenet/visualize_conf_maps/", 
	gt_path = "/home/xxh/Documents/Journal/dataset/sum_train_test_only/test_gt/test4_gt_128_dilate/", num_samples = 400)

compare_MAE(att_path = "/home/xxh/Documents/Journal/code/2densenet/visualize_conf_maps/", 
	gt_path = "/home/xxh/Documents/Journal/dataset/sum_train_test_only/test_gt/test4_gt_128_dilate/", num_samples = 400)