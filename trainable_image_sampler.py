"""
This script operates attention guided image deformation (AGD) using the combination of:
customized grids generator (CGG) and structured grids generator (SGG).

The demonstration is shown in the Fig. 4 of the paper.

sampling_grids = lamda * CGG + (1-lamda) * SGG.
When lamda = 1, it is CGG branch only,
when lamda = 0, it is SGG branch only.

The comparison results of CGG + SGG, CGG only, and SGG only are shown in the TABLE II and Fig. 5 of the paper.
"""
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def uniform_grids_1d(grid_size, padding_size):
	"""
	generate the uniform coords along two dimensions, respectively.
	"""
	global_size = grid_size + 2 * padding_size
	uniform_x = np.zeros((1, global_size))
	uniform_y = np.zeros((global_size, 1))

	for i in range(global_size):
		uniform_x[0, i] = (i - padding_size)/(grid_size - 1.0)
		uniform_y[i, 0] = (i - padding_size)/(grid_size - 1.0)

	return uniform_x, uniform_y


def uniform_grids_2d(grid_size, padding_size):
	"""
	generate uniform grids with size (91, 91, 2), each element has the value between -1 and 2.
	"""
	global_size = grid_size + 2 * padding_size
	uniform_coords = np.zeros((global_size, global_size, 2))

	for k in range(2):
		for i in range(global_size):
			for j in range(global_size):
				uniform_coords[i, j, k] = k * (i - padding_size)/(grid_size - 1.0) \
					+ (1.0 - k) * (j - padding_size)/(grid_size - 1.0)

	return uniform_coords


def Gaussian_1d(size, fwhm = 9):
    """ Make a 1d gaussian kernel.
    size is the length of the kernel, fwhm is the effective radius.
    Return: a gaussian matrix of shape: [1, size]
    """
    x = np.arange(0, size, 1, float)
    x0 = size // 2
    return np.exp(-4*np.log(2) * ((x-x0)**2) / fwhm**2)


def Gaussian_2d(size, fwhm = 9):
    """ Make a square gaussian kernel.
    size is the length of a side of the square, fwhm is the effective radius.
    Return: a gaussian matrix of shape: [size, size]
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    x0 = y0 = size // 2

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)



def gauss_conv_1d(_input, axis, padding_size = 30, strides=[1, 1, 1, 1], padding='VALID'):
	"""
	1d convolution with gaussian kernel.
	"""
	gauss_size = 2 * padding_size + 1

	if axis == 'x':		
		gaussian_weights = tf.reshape(tf.convert_to_tensor(
			Gaussian_1d(gauss_size), dtype = tf.float32), [1, gauss_size]) # (1, 61)
	elif axis == 'y':
		gaussian_weights = tf.reshape(tf.convert_to_tensor(
			Gaussian_1d(gauss_size), dtype = tf.float32), [gauss_size, 1]) # (61, 1)

	gauss_kernel = gaussian_weights[:, :, tf.newaxis, tf.newaxis] 

	in_features = int(_input.get_shape()[-1])
	kernel = tf.tile(gauss_kernel, [1, 1, in_features, in_features])
	output = tf.nn.conv2d(_input, kernel, strides, padding)

	return output

def gauss_conv_2d(_input, padding_size = 30, strides=[1, 1, 1, 1], padding='VALID'):
	"""
	convolution with gaussian kernel.
	"""
	gaussian_weights = tf.convert_to_tensor(
		Gaussian_2d(2 * padding_size + 1), dtype = tf.float32) # (61, 61)
	gauss_kernel = gaussian_weights[:, :, tf.newaxis, tf.newaxis] # (61, 61, 1, 1)	

	in_features = int(_input.get_shape()[-1])
	kernel = tf.tile(gauss_kernel, [1, 1, in_features, in_features])
	output = tf.nn.conv2d(_input, kernel, strides, padding)

	return output


def generate_structured_grid(saliency, batch_size, src_size, dst_size, grid_size = 31, padding_size = 30):
	"""
	compute the column-wise and row-wise sampling positions on the src image.
	"""
	global_size = grid_size + 2 * padding_size

	# generate the uniformly distributed grid of the dst image.
	dst_x, dst_y = uniform_grids_1d(grid_size, padding_size)

	dst_x = np.tile(dst_x[np.newaxis, :, :, np.newaxis], [batch_size, 1, 1, 1]) # (8, 1, 91, 1)
	uniform_x = tf.convert_to_tensor(dst_x, dtype = tf.float32)

	dst_y = np.tile(dst_y[np.newaxis, :, :, np.newaxis], [batch_size, 1, 1, 1]) # (8, 91, 1, 1)
	uniform_y = tf.convert_to_tensor(dst_y, dtype = tf.float32)
	

	# generate the saliency distribution along the x axis.
	saliency_x = tf.reduce_max(saliency, axis = 1, keepdims = True) # (8, 1, 91, 1)
	denominator_x = gauss_conv_1d(saliency_x, axis = 'x') # (8, 1, 31, 1)
	numerator_x = gauss_conv_1d(saliency_x * uniform_x, axis = 'x')
	src_xgrids = numerator_x/denominator_x # (8, 1, 31, 1)

	# generate the saliency distribution along the y axis.
	saliency_y = tf.reduce_max(saliency, axis = 2, keepdims = True) # (8, 91, 1, 1)
	denominator_y = gauss_conv_1d(saliency_y, axis = 'y') # (8, 31, 1, 1)
	numerator_y = gauss_conv_1d(saliency_y * uniform_y, axis = 'y')
	src_ygrids = numerator_y/denominator_y  # (8, 31, 1, 1)


	src_xgrids = tf.clip_by_value(src_xgrids, 0, 1) # (8, 1, 31, 1)
	src_ygrids = tf.clip_by_value(src_ygrids, 0, 1) # (8, 31, 1, 1)

	src_xgrids = src_xgrids * (src_size - 2)
	src_ygrids = src_ygrids * (src_size - 2)

	src_xgrids = tf.tile(src_xgrids, [1, 31, 1, 1])
	src_ygrids = tf.tile(src_ygrids, [1, 1, 31, 1]) # (8, 31, 31, 1)

	# generate the 31*31 sampling grids to resample the feature maps, and further achieves transformation equivariance.
	sample_grids = tf.concat([src_xgrids, src_ygrids], axis = -1) # (8, 31, 31, 2)
	src_grids = tf.image.resize_images(sample_grids, (dst_size, dst_size)) # (8, 128, 128, 2)

	# src_xgrids = tf.image.resize_images(src_xgrids, (dst_size, dst_size)) # (8, 128, 128, 1)
	# src_ygrids = tf.image.resize_images(src_ygrids, (dst_size, dst_size)) # (8, 128, 128, 1)

	# src_grids = tf.concat([src_xgrids, src_ygrids], axis = -1) # (8, 128, 128, 2)

	return sample_grids, src_grids


def generate_pixel_grid(saliency, batch_size, src_size, dst_size, grid_size = 31, padding_size = 30):
	"""
	The CGG,
	compute the pixel-wise sampling positions on the src image.
	"""
	global_size = grid_size + 2 * padding_size

	dst_coords = np.zeros((1, global_size, global_size, 2))
	dst_coords[0, :, :, :] = uniform_grids_2d(grid_size, padding_size)
	dst_coords = np.tile(dst_coords, [batch_size, 1, 1, 1]) # (8, 91, 91, 2)

	dst_coords = tf.convert_to_tensor(dst_coords, dtype = tf.float32)

	denominator = gauss_conv_2d(saliency) # (8, 31, 31, 1)

	uniform_x = tf.reshape(dst_coords[:, :, :, 0], [-1, global_size, global_size, 1])
	uniform_y = tf.reshape(dst_coords[:, :, :, 1], [-1, global_size, global_size, 1])

	numerator_x = gauss_conv_2d(saliency * uniform_x)
	numerator_y = gauss_conv_2d(saliency * uniform_y)

	src_xgrids = numerator_x/denominator
	src_ygrids = numerator_y/denominator

	src_xgrids = tf.clip_by_value(src_xgrids, 0, 1)
	src_ygrids = tf.clip_by_value(src_ygrids, 0, 1) # (8, 31, 31, 1)

	src_xgrids = src_xgrids * (src_size - 2)
	src_ygrids = src_ygrids * (src_size - 2)

	sample_grids = tf.concat([src_xgrids, src_ygrids], axis = -1) # (8, 31, 31, 2)
	src_grids = tf.image.resize_images(sample_grids, (dst_size, dst_size)) # the last dimension denotes x and y.

	return sample_grids, src_grids


def get_resampled_images(image, saliency, batch_size, src_size, dst_size, padding_size = 30, lamda = 0.5):
	"""
	Perform sampling at the specific coordinates in the source images.
	image: (8, 512, 512, 3)
	saliency: (8, 31, 31, 1)
	"""
	s_size = int(saliency.get_shape()[1])
	new_size = s_size + 2 * padding_size

	for i in range(padding_size):
		saliency = tf.pad(saliency, [[0,0],[1,1],[1,1],[0,0]], 'symmetric')

	padded_saliency = saliency
	padded_saliency.set_shape([batch_size, new_size, new_size, 1])

	# generate the sampling grids in the source images.
	structured_sample_grids, structured_src_grids = generate_structured_grid(padded_saliency, batch_size, src_size, dst_size) # (8, 128, 128, 2)
	pixel_wise_sample_grids, pixel_wise_src_grids = generate_pixel_grid(padded_saliency, batch_size, src_size, dst_size) # (8, 128, 128, 2)

	sample_grids = (1.0 - lamda) * structured_sample_grids + lamda * pixel_wise_sample_grids
	src_grids = (1.0 - lamda) * structured_src_grids + lamda * pixel_wise_src_grids

	resampled_image = tf.contrib.resampler.resampler(image, src_grids)

	return resampled_image
