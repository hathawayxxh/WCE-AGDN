"""
Using three metrics to evaluate the attention maps (should be saved as RGB images, rather than heatmaps).
1. precision-recall curves.
2. F-measure.
3. Mean absolute error.
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def prec_rec(att_path, gt_path, num_samples):
	"""
	Binarize the saliency map with a threshold sliding from 0 to 255
	and then compare the binary maps with the ground truth.
	"""
	precision = np.zeros(255)
	recall = np.zeros(255)

	num_imgs = 0
	for split in range(4):
		split_ind = split + 1
		for img in range(num_samples):
			num_imgs += 1
			img += 200
			# print('split' + str(split_ind) + '_' + str(img) + '.jpg')
			att_map = cv2.imread(os.path.join(att_path, 'split' + str(split_ind) + '_' + str(img) + '.jpg'), 0)
			att_map = 255.0 * (att_map - np.min(att_map))/(np.max(att_map) - np.min(att_map))

			gt_map = cv2.imread(os.path.join(gt_path, 'split' + str(split_ind) + '_' + str(img) + '.jpg'), 0)
			gt_map = np.where(gt_map == 0, np.zeros_like(gt_map), np.ones_like(gt_map))

			for iter in range(255):
				pred = np.where(att_map >= iter, np.ones_like(att_map), np.zeros_like(att_map))

				prec = float(np.sum(pred * gt_map))/np.sum(pred + 1e-10)
				rec = float(np.sum(pred * gt_map))/np.sum(gt_map + 1e-10)

				precision[iter] += prec
				recall[iter] += rec

	precision = precision/(num_imgs)
	recall = recall/(num_imgs)

	return recall, precision


def plot_pr_curve(att_path, gt_path, num_samples):

	recall1, precision1 = prec_rec(att_path + "3TLFA_DML_finetune", gt_path, num_samples)
	# recall2, precision2 = prec_rec(att_path, gt_path, num_samples)

	plt.plot(recall1, precision1, color = "green", label = "3TLFA_DML_finetune")
	# plt.plot(recall2, precision2, color = "red", label = "split1_gradcam_att1")

	plt.legend()
	# plt.title("Bleeding_200images")

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim((0.0, 1.0))
	plt.ylim((0.0, 1.0))

	plt.savefig("/home/codes/AGDNet/save_pr_curves/3TLFA_DML_finetune.jpg", bbox_inches='tight')


def F_measure(att_path, gt_path, num_samples, mode):
	precision = 0.0
	recall = 0.0

	num_imgs = 0
	for split in range(4):
		split_ind = split + 1
		for img in range(num_samples):
			num_imgs += 1
			img += 200
			att_map = cv2.imread(os.path.join(att_path, 'split' + str(split_ind) + '_' + str(img) + '.jpg'), 0)
			# att_map = 255.0 * (att_map - np.min(att_map) + 1e-10)/(np.max(att_map) - np.min(att_map) + 1e-10)
			att_map = 255.0 * (att_map - np.min(att_map)) / (np.max(att_map) - np.min(att_map))
			threshold = 2 * np.mean(att_map)
			pred = np.where(att_map >= threshold, np.ones_like(att_map), np.zeros_like(att_map))

			gt_map = cv2.imread(os.path.join(gt_path, 'split' + str(split_ind) + '_' + str(img) + '.jpg'), 0)
			gt_map = np.where(gt_map == 0, np.zeros_like(gt_map), np.ones_like(gt_map))

			precision += float(np.sum(pred * gt_map))/np.sum(pred + 1e-10)
			recall += float(np.sum(pred * gt_map))/np.sum(gt_map + 1e-10)

	precision = precision/(num_imgs)
	recall = recall/(num_imgs)

	beta_2 = 4.0
	F_beta = (1 + beta_2) * precision * recall/(beta_2 * precision + recall + 1e-10)

	print (mode, "Precision:", precision, "Recall:", recall, "F_measure:", F_beta)




def compare_F_measure(att_path, gt_path, num_samples):

	print ("Attention F1_measure")
	# F_measure(att_path + "split1_B2_saliency", gt_path, num_samples, mode = "avg_net1")
	F_measure(att_path + "3TLFA_DML_finetune", gt_path, num_samples, mode = "3TLFA_DML_finetune")


def MAE_measure(att_path, gt_path, num_samples, mode):
	"""
	Mean absolute error.
	Average difference between the prediction and ground truth in pixel level.
	"""
	MAE_error = 0.0
	num_imgs = 0
	for split in range(4):
		split_ind = split + 1

		for img in range(num_samples):
			num_imgs += 1
			img += 200
			att_map = cv2.imread(os.path.join(att_path, 'split' + str(split_ind) + '_' + str(img) + '.jpg'), 0)
			pred = (att_map - np.min(att_map))/(np.max(att_map) - np.min(att_map)).astype(float)

			gt_map = cv2.imread(os.path.join(gt_path, 'split' + str(split_ind) + '_' + str(img) + '.jpg'), 0)
			gt_map = np.where(gt_map == 0, np.zeros_like(gt_map), np.ones_like(gt_map)).astype(float)

			MAE_error += np.mean(np.abs(pred - gt_map))

	MAE_error = MAE_error/(num_imgs)
	print (mode, "MAE_error:", MAE_error)



def compare_MAE(att_path, gt_path, num_samples):

	print ("Mean absolute error:")
	# MAE_measure(att_path + "split1_B2_saliency", gt_path, num_samples, mode = "avg_net1")
	MAE_measure(att_path + "3TLFA_DML_finetune", gt_path, num_samples, mode = "3TLFA_DML_finetune")


#####################################
### Run functions
#####################################
plot_pr_curve(att_path = "/home/codes/AGDNet/visualize_attention/", gt_path = "/home/datasets/images/test_gt_sum/", num_samples = 400)

compare_F_measure(att_path = "/home/codes/AGDNet/visualize_attention/", gt_path = "/home/datasets/images/test_gt_sum/", num_samples = 400)

compare_MAE(att_path = "/home/codes/AGDNet/visualize_attention/", gt_path = "/home/datasets/images/test_gt_sum/", num_samples = 400)