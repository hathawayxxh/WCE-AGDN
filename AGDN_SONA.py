"""
Replace the 3 DTOA modules with second-order non-local (SONA) modules.
"""

import os
import cv2
import time
import shutil
from ops import *
from ops_guo import *
from skimage import io
from datetime import timedelta
import tensorflow.contrib.slim as slim

import numpy as np
import tensorflow as tf
import  pandas as pd # used to write and read csv files.
from skimage import io, data, color
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import high_low_res_data_provider
from high_low_res_data_provider import get_image_label_batch

import trainable_image_sampler
from trainable_image_sampler import get_resampled_images

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class DenseNet:
    def __init__(self, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=0.5,
                 bc_mode=True,
                 **kwargs):

        self.data_shape = (128, 128, 3)
        self.n_classes = 3
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        # the depth is consisted of layers in dense blocks and layers in transition layers.
        # the number of layers in different blocks.
        # self.layers_per_block = [6,12,24,16]
        self.layers_per_block = [4, 8, 12, 8]
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "totally %d layers." % (
                      model_type, self.total_blocks, self.depth))
        if bc_mode:
            # the layers in each block is consisted of bottleneck layers and composite layers,
            # so the number of composite layers should be half the total number.
            print("Build %s model with %d blocks, "
                  "totally %d layers." % (
                      model_type, self.total_blocks, self.depth))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        # self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0
        self.batch_size = 8

        self.num_train = 9688
        self.num_test = 600

        # self.num_train = 800
        self.num_test = 8
        self.src_size = 320
        self.dst_size = 128

        self.margin = 0.05

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    # if the save_path exists, use the save path
    # else create a save path
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.chkpt-80')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}".format(
            self.model_type, self.growth_rate, self.depth)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)

        # load_fn = slim.assign_from_checkpoint_fn(
        #     self.save_path, tf.global_variables(),ignore_missing_vars=True)
        # load_fn(self.sess)        

        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss1, loss2, loss3, TE_loss, accuracy1, accuracy2, accuracy, 
                          epoch, prefix, should_print=True):
        if should_print:
            print("mean cross_entropy1: %f, mean accuracy1: %f" % (loss1, accuracy1))
            print('')
            print("mean cross_entropy2: %f, mean accuracy2: %f" % (loss2, accuracy2))
            print('')
            print("mean loss in sum branch: %f, mean accuracy: %f" % (loss3, accuracy))
            print('')
            print("mean TE loss: %f" % (TE_loss))

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='loss1_%s' % prefix, simple_value=float(loss1)),
            tf.Summary.Value(tag='loss2_%s' % prefix, simple_value=float(loss2)),
            tf.Summary.Value(tag='accuracy1_%s' % prefix, simple_value=float(accuracy1)),
            tf.Summary.Value(tag='accuracy2_%s' % prefix, simple_value=float(accuracy2))
        ])

        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='images_input1')

        self.high_res_images = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.src_size, self.src_size, 3],
            name='high_resolution_input_images')

        self.labels = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.n_classes],
            name='labels')

        self.gt_map = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.dst_size, self.dst_size, 1],
            name='gt_map')

        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')

        self.is_training = tf.placeholder(tf.bool, shape=[])


    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        # the function batch_norm, conv2d, dropout are defined in the following part.
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)

            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, block, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block[block]):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and then average
        pooling
        """
        # call composite function with 1x1 kernel
        # reduce the number of feature maps by compression
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output


    def nonlocal_block(self, x, channels, de=4, scope='Nonlocal', trainable=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):        

            f = conv(x, channels // de, kernel=1, stride=1, sn=False, scope='f_conv')
            g = conv(x, channels // de, kernel=1, stride=1, sn=False, scope='g_conv')
            h = conv(x, channels, kernel=1, stride=1, sn=False, scope='h_conv')

            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta_a = tf.nn.softmax(s, dim=-1)  # attention map

            o = tf.matmul(beta_a, hw_flatten(h)) # [bs, N, C]
            
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            x = gamma * o + x
            print ("Nonlocal output:", x)
        return x

    def SONA(self, x, channels, de=4, scope='SONA', trainable=True, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):  

            c_num = channels // de
            f = conv(x, c_num, kernel=1, stride=1, sn=False, scope='f_conv')
            f = tf.nn.relu(f) #[bs, h, w, c_num]

            h = conv(x, channels, kernel=1, stride=1, sn=False, scope='h_conv') #[bs, h, w, c_num]

            ### compute the spatial correlation.        
            I_hat = (-1./c_num/c_num) * tf.ones([c_num, c_num]) + (1./c_num) * tf.eye(c_num)
            I_hat = tf.tile(tf.expand_dims(I_hat, 0), [x.shape[0], 1, 1]) # # (8, c, c)
            correlation = tf.matmul(tf.matmul(hw_flatten(f), I_hat), hw_flatten(f), False, True)
            # correlation = tf.nn.softmax(correlation/(tf.sqrt(tf.cast(c_num, tf.float32))), axis = -1) # (8, hw, hw)
            correlation = tf.nn.softmax(correlation, axis = -1)

            ### aggregated feature
            o = tf.matmul(correlation, hw_flatten(h)) # [bs, hw, C]
            
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            output = gamma * o + x

            print ("SONA output:", output)
        return output


    # after block4, convert the 7*7 feature map to 1*1 by average pooling.
    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        spatial_features = output

        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        print(output.shape)
        # FC

        features = tf.reshape(output, [self.batch_size, -1]) 

        with tf.variable_scope("final_layer") as scope:
            output = self.conv2d(output, out_features = 3, kernel_size = 1)

            scope.reuse_variables()
            spatial_output = self.conv2d(spatial_features, out_features = 3, kernel_size = 1)
            spatial_pred = tf.nn.softmax(spatial_output) # (16, 14, 14, 3)
            # spatial_pred = tf.reshape(spatial_pred, [self.batch_size, -1, 3]) # (16, 196, 3)
        print output.shape, spatial_pred.shape

        logits = tf.reshape(output, [-1, self.n_classes])
        print features.shape, logits.shape

        return features, logits, spatial_pred


    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output


    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        # output = tf.contrib.layers.batch_norm(
        #     _input, scale=True, is_training=self.is_training,
        #     updates_collections=None)
        output = tf.contrib.layers.batch_norm(
            _input, decay = 0.9, epsilon = 1e-05, 
            center = True, scale=True, is_training=self.is_training,
            updates_collections=None)

        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output


    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())
            # an initializer that generates tensors with unit variance.

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)


    def rank_loss(self, pred1, pred2):
        # The ranking loss between two branches,
        # if prob2 > prob1 + margin, rank_loss = 0
        # if prob2 < prob1 + margin, rank_loss = prob1 - prob2 + margin
        prob1 = tf.reduce_sum(tf.multiply(self.labels, pred1), axis = 1) # (batch_size, 1)
        prob2 = tf.reduce_sum(tf.multiply(self.labels, pred2), axis = 1) # (batch_size, 1)
        rank_loss = tf.reduce_mean(tf.maximum(0.0, prob1 - prob2 + self.margin)) # scalar
        return rank_loss


    def spatial_kl_loss(self, resampled_s_map1, s_map2):
        """
        Compute the KL divergence between the attention values in each positions,
        and the average KL loss.
        """
        resampled_s_map1 = tf.reshape(resampled_s_map1, [self.batch_size, -1]) # (8, 64)
        s_map2 = tf.reshape(s_map2, [self.batch_size, -1]) # (8, 64)

        kl_loss = -tf.reduce_mean(s_map2 * tf.log(resampled_s_map1))
        return kl_loss


    def compute_saliency(self, f_maps, mode = "avg"):

        if mode == "avg":
            f_maps = tf.nn.relu(f_maps)
            s_map = tf.reduce_mean(f_maps, axis = -1, keepdims = True)
        elif mode == "max":
            f_maps = tf.nn.relu(f_maps)
            s_map = tf.reduce_max(f_maps, axis = -1, keepdims = True)
        elif mode == "sum_abs":
            f_maps = tf.abs(f_maps)
            s_map = tf.reduce_sum(f_maps, axis = -1, keepdims = True)

        s_map_min = tf.reduce_min(s_map, axis = [1, 2, 3], keepdims = True) # [batch_size, 8, 8, 1]
        s_map_max = tf.reduce_max(s_map, axis = [1, 2, 3], keepdims = True)
        s_map = tf.div(s_map - s_map_min + 1e-8, s_map_max - s_map_min + 1e-8) # [batch_size, 8, 8, 1]
        # s_map = tf.div(s_map, s_map_max)

        # s_map = tf.sigmoid(s_map)

        s_map = tf.image.resize_images(s_map, size = (31, 31)) # used for image resampling.

        saliency_map = tf.tile(s_map, (1,1,1,3))
        saliency_map = tf.image.resize_images(saliency_map, (128, 128)) # (8, 128, 128, 3)

        return s_map, saliency_map


    def generate_pseudo_gt(self, att_map):
        """
        Convert the saliency maps of net2 into pseudo ground truth labels.
        """
        # f_maps = tf.nn.relu(f_maps)
        # att_map = tf.reduce_mean(f_maps, axis = -1, keepdims = True)
        threshold = 0.6 * tf.reduce_max(att_map, axis = (1,2,3), keepdims = True)
        # threshold = tf.reduce_mean(att_map, axis = (1,2,3), keepdims = True)

        fg = tf.ones_like(att_map)
        bg = tf.zeros_like(att_map)

        att_mask = tf.where(att_map > threshold, x = fg, y = bg)

        mask = tf.tile(att_mask, (1,1,1,3))
        mask = tf.image.resize_images(mask, (128, 128))

        return att_mask, mask


    def select_samples(self, pred1, pred2, mode = "net2_teach_net1"):
        """
        Select the samples that pred2 are more accurate than pred1,
        and also, pred2 make correct decision,
        then let the resampled att_maps1 of these samples be similar with att_maps2.
        """
        prob1 = tf.reduce_sum(self.labels * pred1, axis = 1)
        prob2 = tf.reduce_sum(self.labels * pred2, axis = 1)
        threshold = 0.5

        cond1 = tf.cast(tf.greater(prob2, tf.maximum(prob1, threshold)), tf.float32)
        cond2 = tf.cast(tf.equal(tf.argmax(pred2, 1), tf.argmax(self.labels, 1)), tf.float32)
        cond3 = tf.cast(tf.greater(tf.argmax(self.labels, 1), 0), tf.float32)

        if mode == "net2_teach_net1":
            select = cond3
        elif mode == "net1_teach_net2":
            select = tf.multiply(tf.multiply(cond1, cond2), cond3)

        return select


    def network(self, _input):
        """
        Define the structure of the backbone network, 
        this network will be used in both two branches.
        Input: image,
        Return: the feature vector of input image(batch_size, channels), 
        predicted logits(batch_size, 3), predicted spatial probability(batch_size, h, w, 3).
        """
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block

        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                _input,
                out_features=self.first_output_features,
                kernel_size=3, strides = [1, 1, 1, 1])
            print(output.shape)

        with tf.variable_scope("Initial_pooling"):
            output = tf.nn.max_pool(output, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
            print(output.shape)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(block, output, growth_rate, layers_per_block)
            print(output.shape)

            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)
                    print(output.shape)

            if block == 0:
                output = self.SONA(output, int(output.shape[-1]), de=4, scope='SONA1', trainable=True, reuse=False)
                fmaps_b1 = output

            if block == 1:
                output = self.SONA(output, int(output.shape[-1]), de=4, scope='SONA2', trainable=True, reuse=False)
                fmaps_b2 = output

            if block == 2:
                output = self.SONA(output, int(output.shape[-1]), de=4, scope='SONA3', trainable=True, reuse=False)
                fmaps_b3 = output
                
        f_maps = output

        # the last block is followed by a "transition_to_classes" layer.
        with tf.variable_scope("Transition_to_classes"):
            features, logits, spatial_pred = self.transition_layer_to_classes(f_maps)

        return f_maps, fmaps_b1, fmaps_b2, fmaps_b3, features, logits, spatial_pred



    def _build_graph(self):
              
        with tf.variable_scope("net1") as scope:
            f_maps1, fmaps1_b1, fmaps1_b2, fmaps1_b3, self.features1, logits1, spatial_pred1 = self.network(self.images)
            self.pred1 = tf.nn.softmax(logits1)
            self.f_maps1 = tf.nn.relu(f_maps1)

            self.s_map1, self.saliency1 = self.compute_saliency(f_maps1)
            _, self.smaps1_b1 = self.compute_saliency(fmaps1_b1)
            _, self.smaps1_b2 = self.compute_saliency(fmaps1_b2)
            _, self.smaps1_b3 = self.compute_saliency(fmaps1_b3)

        self.input2 = tf.stop_gradient(get_resampled_images(
            self.high_res_images, self.s_map1, self.batch_size, self.src_size, self.dst_size))

        # self.input2 = self.images

        self.resampled_s_map1 = get_resampled_images(self.s_map1, self.s_map1, self.batch_size, 31, 8)
        print (self.resampled_s_map1.shape)

        self.resampled_gt = get_resampled_images(
            self.gt_map, self.s_map1, self.batch_size, self.dst_size, self.dst_size)

        resampled_saliency1 = tf.tile(self.resampled_s_map1, (1,1,1,3))
        self.resampled_saliency1 = tf.image.resize_images(resampled_saliency1, (128, 128)) # (8, 128, 128, 3)
        self.resampled_seg_map1 = generate_seg(self.resampled_saliency1)

        with tf.variable_scope("net2") as scope:
            f_maps2, fmaps2_b1, fmaps2_b2, fmaps2_b3, self.features2, logits2, spatial_pred2 = self.network(self.input2)
            self.pred2 = tf.nn.softmax(logits2)

            # Saliency of the original features of block4 and the SACA feature.
            self.s_map2, self.saliency2 = self.compute_saliency(f_maps2)
            self.s_map2 = tf.image.resize_images(self.s_map2, [8, 8]) # used for TE-loss computation.


        ##########################################################
        # The 3-rd branch: combine the features from net1 and net2, 
        # and, then make predictions of the combined features.
        ##########################################################

        total_fmaps = tf.concat(axis = 3, values = (f_maps1, f_maps2))

        with tf.variable_scope("Sum_branch") as scope:
            _, logits, _ = self.transition_layer_to_classes(total_fmaps)
            self.pred = tf.nn.softmax(logits)


        # weighted loss:
        class_weights = tf.constant([1, 1, 1])
        weights = tf.gather(class_weights, tf.argmax(self.labels, axis = -1))
        cross_entropy1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits1, weights = weights, label_smoothing = 0.1))
        self.cross_entropy1 = cross_entropy1

        self.kd_loss1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.pred, logits1, weights = weights, label_smoothing = 0.1))

        cross_entropy2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits2, weights = weights, label_smoothing = 0.1))
        self.cross_entropy2 = cross_entropy2

        self.sum_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits, weights = weights, label_smoothing = 0.1))

        self.grads = tf.gradients(self.cross_entropy2, self.images)[0]
        print('gradients from loss2 to input1:', self.grads)

        self.rank_loss_2_1 = self.rank_loss(self.pred1, self.pred2) # force the pred2 more accurate than pred1

        rank_grad = tf.gradients(self.rank_loss_2_1, self.input2)[0]
        print ("gradients from rank_loss to input2:", rank_grad)

        self.TE_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
            tf.square(self.resampled_s_map1 - self.s_map2), axis = [1, 2, 3])))
        # self.TE_loss = self.spatial_kl_loss(self.resampled_s_map1, self.s_map2)

        #### Selective MSE loss
        self.net2_teach_net1 = self.select_samples(self.pred1, self.pred2, mode = "net2_teach_net1")
        self.TE_loss1 = tf.reduce_mean(self.net2_teach_net1 * tf.sqrt(tf.reduce_sum(
            tf.square(self.resampled_s_map1 - self.s_map2), axis = [1, 2, 3])))

        self.net1_teach_net2 = self.select_samples(self.pred2, self.pred1, mode = "net1_teach_net2")
        self.TE_loss2 = tf.reduce_mean(self.net1_teach_net2 * tf.sqrt(tf.reduce_sum(
            tf.square(self.resampled_s_map1 - self.s_map2), axis = [1, 2, 3])))


        self.grad1 = tf.gradients(self.sum_loss, self.images)[0]
        print ("gradients from sum loss to input1:", self.grad1)

        self.grad2 = tf.gradients(self.sum_loss, self.input2)[0]
        print ("gradients from sum loss to input2:", self.grad2)

        # regularize the variables that needs to be trained in Net1 or Net2.
        var_list = [var for var in tf.trainable_variables()]
        var_list1 = [var for var in tf.trainable_variables() if var.name.split('/')[0] == 'net1']
        var_list2 = [var for var in tf.trainable_variables() if var.name.split('/')[0] == 'net2']
        var_list3 = [var for var in tf.trainable_variables() if var.name.split('_')[0] == 'Sum']
        # print var_list3
        # print (var_list1 + var_list2)

        # print ("--------------------", len(var_list))
        # for var in tf.trainable_variables():
        #     print var.name.split('/')[0]

        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        l2_loss1 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('/')[0] == 'net1'])
        l2_loss2 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('/')[0] == 'net2'])
        l2_loss3 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('_')[0] == 'Sum'])


        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)


        self.train_step1 = optimizer.minimize(
            self.cross_entropy1 + l2_loss1 * self.weight_decay, var_list = var_list1) # + 0.5 * self.TE_loss1
        self.train_step2 = optimizer.minimize(
            self.cross_entropy2 + l2_loss2 * self.weight_decay, var_list = var_list2) # + 0.5 * self.TE_loss2

        self.train_step3 = optimizer.minimize(
            self.sum_loss + l2_loss3 * self.weight_decay, var_list = var_list3)        


        correct_prediction1 = tf.equal(
            tf.argmax(self.pred1, 1),
            tf.argmax(self.labels, 1))
        self.correct_prediction1 = correct_prediction1
        self.accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
        # self.precision1, self.recall1, self.F1_1 = self.net_measure(prediction1, self.labels)
        # self.precision1 = tf.convert_to_tensor(self.precision1)

        correct_prediction2 = tf.equal(
            tf.argmax(self.pred2, 1),
            tf.argmax(self.labels, 1))
        self.correct_prediction2 = correct_prediction2
        self.accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
        # self.precision2, self.recall2, self.F1_2 = self.net_measure(prediction2, self.labels)

        self.correct_prediction = tf.equal(
            tf.argmax(self.pred, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))        


    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        # self.batch_size = batch_size
        # reduce the lr at epoch1 and epoch2.
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        reduce_lr_epoch_3 = train_params['reduce_lr_epoch_3']
        reduce_lr_epoch_4 = train_params['reduce_lr_epoch_4']

        total_start_time = time.time()

        loss1_all_epochs = []
        loss2_all_epochs = []
        loss3_all_epochs = []

        acc1_all_epochs = []
        acc2_all_epochs = []
        acc_all_epochs = []

        nr1_all_epochs = []
        br1_all_epochs = []
        ir1_all_epochs = []
        kappa1_all_epochs = []

        nr2_all_epochs = []
        br2_all_epochs = []
        ir2_all_epochs = []
        kappa2_all_epochs = []

        nr_all_epochs = []
        br_all_epochs = []
        ir_all_epochs = [] 
        kappa_all_epochs = []

        """
        Only save the model with highest accuracy2
        """
        best_acc = 0.0

        self.train_high_image_batch, self.train_low_image_batch, self.train_label_batch = \
                                    get_image_label_batch(batch_size, shuffle=True, name='train')
        self.test_high_image_batch, self.test_low_image_batch, self.test_label_batch = \
                                    get_image_label_batch(batch_size, shuffle=False, name='tet_3samples')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord = coord)

        try:

            for epoch in range(1, n_epochs + 1):
                start_time = time.time()

                # print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
                # if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2 \
                # or epoch == reduce_lr_epoch_3 or epoch == reduce_lr_epoch_4:
                #     learning_rate = learning_rate / 10
                #     print("Decrease learning rate, new lr = %f" % learning_rate)

                # print("Training...")
                # loss1, loss2, loss3, TE_loss, rank_loss, acc1, acc2 = self.train_one_epoch(batch_size, learning_rate)
                # if self.should_save_logs:
                #     self.log_loss_accuracy(loss1, loss2, loss3, TE_loss, acc1, acc2, acc2, epoch, prefix='train')

                # print ("Rank loss:", rank_loss)

                # loss1_all_epochs.append(loss1)
                # loss2_all_epochs.append(loss2)
                # loss3_all_epochs.append(loss3)

                if train_params.get('validation_set', False):
                    print("Validation...")
                    loss1, loss2, loss3, TE_loss, acc1, acc2, acc, nr1, br1, ir1, kappa1, \
                        nr2, br2, ir2, kappa2, nr, br, ir, kappa = self.test(batch_size)

                    if self.should_save_logs:
                        self.log_loss_accuracy(loss1, loss2, loss3, TE_loss, acc1, acc2, acc, epoch, prefix='valid')


                    acc1_all_epochs.append(acc1)
                    acc2_all_epochs.append(acc2)
                    acc_all_epochs.append(acc)

                    nr1_all_epochs.append(nr1)
                    br1_all_epochs.append(br1)
                    ir1_all_epochs.append(ir1)
                    kappa1_all_epochs.append(kappa1)

                    nr2_all_epochs.append(nr2)
                    br2_all_epochs.append(br2)
                    ir2_all_epochs.append(ir2)
                    kappa2_all_epochs.append(kappa2)

                    nr_all_epochs.append(nr)
                    br_all_epochs.append(br)
                    ir_all_epochs.append(ir)
                    kappa_all_epochs.append(kappa)


                time_per_epoch = time.time() - start_time
                seconds_left = int((n_epochs - epoch) * time_per_epoch)
                print("Time per epoch: %s, Est. complete in: %s" % (
                    str(timedelta(seconds=time_per_epoch)),
                    str(timedelta(seconds=seconds_left))))

                if self.should_save_model:
                    if epoch >= 60 and epoch % 5 == 0: #
                        self.save_model(global_step = epoch)

                    if acc >= best_acc:
                        best_acc = acc
                        self.save_model(global_step = 1)

                    # self.save_model(global_step = 1)


            dataframe = pd.DataFrame({'train_loss1': loss1_all_epochs, 'train_loss2': loss2_all_epochs, 'train_loss3': loss3_all_epochs, 'accuracy1': acc1_all_epochs, 
                'accuracy2': acc2_all_epochs, 'accuracy': acc_all_epochs, 'normal_recall_1': nr1_all_epochs, 'normal_recall_2': nr2_all_epochs,
                'normal_recall': nr_all_epochs, 'bleed_recall_1': br1_all_epochs, 'bleed_recall_2': br2_all_epochs, 'bleed_recall': br_all_epochs, 
                'inflam_recall_1': ir1_all_epochs, 'inflam_recall_2': ir2_all_epochs, 'inflam_recall': ir_all_epochs, 
                'kappa1': kappa1_all_epochs, 'kappa2': kappa2_all_epochs, 'kappa': kappa_all_epochs,})

            dataframe.to_csv("./acc_results/result.csv", index = True, sep = ',')

            total_training_time = time.time() - total_start_time
            print("\nTotal training time: %s" % str(timedelta(
                seconds=total_training_time)))

        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
            coord.join(threads)


    def train_one_epoch(self, batch_size, learning_rate):

        train_features_path = "./feature_visualization/train_features.txt" 
        train_labels_path = "./feature_visualization/train_labels.txt"

        total_loss1 = []
        total_loss2 = []
        total_loss3 = []
        total_TE_loss = []
        total_rank_loss = []

        total_prob1 = []
        total_prob2 = []

        total_pred1 = []
        total_pred2 = []
        total_labels1 = []

        for i in range(self.num_train // batch_size):
            high_images, low_images, labels = self.sess.run([
                self.train_high_image_batch, self.train_low_image_batch, self.train_label_batch])

            # the class_labels for features in Net1 are 0,1,2
            class_labels1 = np.argmax(labels, axis = 1).astype(np.int32)
            # the class_labels for features in Net2 are 3,4,5
            class_labels2 = class_labels1 + 3

            feed_dict = {
                self.images: low_images,
                self.high_res_images: high_images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }


            fetches = [self.train_step1, self.train_step2, self.train_step3, self.features1, self.features2, 
                        self.cross_entropy1, self.cross_entropy2, self.sum_loss, self.TE_loss, self.rank_loss_2_1, \
                        self.accuracy1, self.accuracy2, self.pred1, self.pred2] # , self.train_step3

            results = self.sess.run(fetches, feed_dict=feed_dict)
            _, _, _, features1, features2, loss1, loss2, loss3, TE_loss, rank_loss, acc1, acc2, pred1, pred2 = results
            

            features = np.vstack((features1, features2))
            class_labels = np.hstack((class_labels1, class_labels2))
            # print features.shape, class_labels.shape

            if i == 0:
                total_features = features
                total_labels = class_labels
            else:
                total_features = np.append(total_features, features, axis = 0)
                total_labels = np.append(total_labels, class_labels)

            # print(pred)
            total_loss1.append(loss1)
            total_loss2.append(loss2)
            total_loss3.append(loss3)
            total_TE_loss.append(TE_loss)
            total_rank_loss.append(rank_loss)

            prob1 = np.sum(np.multiply(pred1, labels), axis = 1) #[batch_size]
            prob2 = np.sum(np.multiply(pred2, labels), axis = 1)

            total_prob1.append(prob1)
            total_prob2.append(prob2)

            total_pred1.append(np.argmax(pred1, axis = 1))
            total_pred2.append(np.argmax(pred2, axis = 1))
            total_labels1.append(np.argmax(labels, axis = 1))

            if self.should_save_logs:
                self.batches_step += 1
                # save loss and accuracy into Summary
                self.log_loss_accuracy(
                    loss1, loss2, loss3, TE_loss, acc1, acc2, acc2, self.batches_step, 
                    prefix='per_batch', should_print=False)

        mean_loss1 = np.mean(total_loss1)
        mean_loss2 = np.mean(total_loss2)
        mean_loss3 = np.mean(total_loss3)
        mean_TE_loss = np.mean(total_TE_loss)
        mean_rank_loss = np.mean(total_rank_loss)

        overall_acc_1, normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1 = get_accuracy(
            preds = total_pred1, labels = total_labels1)
        overall_acc_2, normal_recall_2, bleed_recall_2, inflam_recall_2, kappa_2 = get_accuracy(
            preds = total_pred2, labels = total_labels1)

        total_prob1 = np.reshape(total_prob1, (-1))
        total_prob2 = np.reshape(total_prob2, (-1))
        ratio = superior_net2(total_prob1, total_prob2)
        print ("The ratio of higher prob2 than prob1:", ratio)

        # print total_features.shape, total_labels.shape
        # np.savetxt(train_features_path, total_features)
        # np.savetxt(train_labels_path, total_labels, fmt = "%d")

        return mean_loss1, mean_loss2, mean_loss3, mean_TE_loss, \
            mean_rank_loss, overall_acc_1, overall_acc_2


    def test(self, batch_size):

        input_path = "./visualize_conf_maps/input/"
        # saliency_path = "./visualize_conf_maps/saliency_save/B2 + 3SONA(lr=0.001)/split2/"
        saliency_path = "./visualize_conf_maps/saliency_maps/"
        gt_path = "/home/xxh/Documents/Journal/dataset/test_gt/"
        dst_gt_path = "/home/xxh/Documents/Journal/dataset/test_gt_128/"
        dilate_gt_path = "/home/xxh/Documents/Journal/dataset/test_gt_128_dilate/"
        resampled_gt_path = "/home/xxh/Documents/Journal/dataset/resampled_gt/"

        total_loss1 = []
        total_loss2 = []
        total_loss3 = []
        total_TE_loss = []

        total_prob1 = []
        total_prob2 = []

        total_pred1 = []
        total_pred2 = []
        total_pred = []
        total_labels = []

        ### Measure the relationship between the att_acc and the prob of the correct class.
        lesion_att_acc1 = []
        lesion_att_acc2 = []

        lesion_probs1 = []
        lesion_probs2 = []

        gt_batch = np.zeros((batch_size, 128, 128, 1))

        epsilon = 1e-8

        bleed_feature = np.zeros(201)
        inflam_feature = np.zeros(201)

        for i in range(self.num_test // batch_size):
            test_high_images, test_low_images, test_labels = self.sess.run([
                self.test_high_image_batch, self.test_low_image_batch, self.test_label_batch])

            if i > 24:
                for num in range(batch_size):
                    gt_batch[num] = np.expand_dims(
                        cv2.imread(os.path.join(dst_gt_path, str(i * batch_size + num) + '.jpg'), 0), axis = -1)

            feed_dict = {
                self.images: test_low_images,
                self.high_res_images: test_high_images,
                self.labels: test_labels,
                self.is_training: False,
                self.gt_map: gt_batch,
            }
    

            fetches = [self.input2, self.cross_entropy1, self.cross_entropy2, self.sum_loss, self.TE_loss, self.saliency1, \
                        self.saliency2, self.smaps1_b1, self.smaps1_b2, self.smaps1_b3, self.pred1, self.pred2, self.pred]

            input2, loss1, loss2, loss3, TE_loss, s_map1, s_map2, smap1_b1, smap1_b2, smap1_b3, pred1, pred2, pred = \
                        self.sess.run(fetches, feed_dict=feed_dict)


            total_loss1.append(loss1)
            total_loss2.append(loss2)
            total_loss3.append(loss3)
            total_TE_loss.append(TE_loss)

            prob1 = np.sum(np.multiply(pred1, test_labels), axis = 1) #[batch_size]
            prob2 = np.sum(np.multiply(pred2, test_labels), axis = 1)

            total_prob1.append(prob1)
            total_prob2.append(prob2)

            pred1 = np.argmax(pred1, axis = 1)
            pred2 = np.argmax(pred2, axis = 1)
            pred = np.argmax(pred, axis = 1)
            labels = np.argmax(test_labels, axis = 1)

            total_pred1.append(pred1)
            total_pred2.append(pred2)
            total_pred.append(pred)
            total_labels.append(labels)


            bleed_index = np.expand_dims(np.where(labels == 1, 1, 0), axis = -1)
            inflam_index = np.expand_dims(np.where(labels == 2, 1, 0), axis = -1)
            # bleed_feature += np.sum(bleed_index * f1, axis = 0) # [c_num]
            # inflam_feature += np.sum(inflam_index * f1, axis = 0)


            for index in range(batch_size):
                img_index = i * batch_size + index
                # if labels[index] != 0:                    
                # save_img(test_low_images[index], img_index, input_path, img_name = '_original.jpg', mode = "image")
                # save_img(input2[index], img_index, input_path, img_name = '_resampled.jpg', mode = "image")

                save_img(s_map1[index], img_index, saliency_path, img_name = '_avg_net1.jpg', mode = "image")
                save_img(s_map2[index], img_index, saliency_path, img_name = '_avg_net2.jpg', mode = "image")
                save_img(test_low_images[index], img_index, saliency_path, img_name = '_input1.jpg', mode = "image")
                save_img(input2[index], img_index, saliency_path, img_name = '_input2.jpg', mode = "image")

                # save_img(s_map1[index], img_index, saliency_path, img_name = '_att_b4.jpg', mode = "image")
                # save_img(smap1_b1[index], img_index, saliency_path, img_name = '_att_b1.jpg', mode = "image")
                save_img(smap1_b2[index], img_index, saliency_path, img_name = '_att_b2.jpg', mode = "image")
                save_img(smap1_b3[index], img_index, saliency_path, img_name = '_att_b3.jpg', mode = "image")


        # print pred1, pred2, labels
        mean_loss1 = np.mean(total_loss1)
        mean_loss2 = np.mean(total_loss2)
        mean_loss3 = np.mean(total_loss3)
        mean_TE_loss = np.mean(total_TE_loss)

        overall_acc_1, normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1 = get_accuracy(preds = total_pred1, labels = total_labels)
        overall_acc_2, normal_recall_2, bleed_recall_2, inflam_recall_2, kappa_2 = get_accuracy(preds = total_pred2, labels = total_labels)
        overall_acc, normal_recall, bleed_recall, inflam_recall, kappa = get_accuracy(preds = total_pred, labels = total_labels)

        # print ("========================Network1======================")
        # print_scores(normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1)

        # print ("========================Network2======================")
        # print_scores(normal_recall_2, bleed_recall_2, inflam_recall_2, kappa_2)

        # print ("=================Ensembled performance================")
        # print_scores(normal_recall, bleed_recall, inflam_recall, kappa)

        total_prob1 = np.reshape(total_prob1, (-1))
        total_prob2 = np.reshape(total_prob2, (-1))
        ratio = superior_net2(total_prob1 - 0.0, total_prob2)
        print ("The ratio of higher prob2 than prob1:", ratio)

        false_index1 = np.where(np.equal(np.reshape(total_pred1, (-1)), np.reshape(total_labels, (-1))) == False)[0]
        false_index2 = np.where(np.equal(np.reshape(total_pred2, (-1)), np.reshape(total_labels, (-1))) == False)[0]
        false_index = np.where(np.equal(np.reshape(total_pred, (-1)), np.reshape(total_labels, (-1))) == False)[0]

        # mean_prob1 = np.mean(total_prob1)
        # mean_prob2 = np.mean(total_prob2)
        # print ("The mean prob of net1 and net2 are:", mean_prob1, mean_prob2)

        # print('====================show misclassified images==================')
        # print('the number of misclassified examples in Net1 is:', len(false_index1))
        # print false_index1

        # print('the number of misclassified examples in Net2 is:', len(false_index2))
        # print false_index2

        # total_pred1 = np.reshape(total_pred1, (-1))
        # for i in range(len(false_index1)):  
        #     print (false_index1[i], total_pred1[false_index1[i]])          

        return mean_loss1, mean_loss2, mean_loss3, mean_TE_loss, overall_acc_1, overall_acc_2, overall_acc, \
            normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1, normal_recall_2, bleed_recall_2, \
            inflam_recall_2, kappa_2, normal_recall, bleed_recall, inflam_recall, kappa