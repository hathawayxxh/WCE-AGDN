import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

#############################################
### Attention
#############################################
def spatial_second_order_att(feature_maps):
    """
    Compute the correlation between different positions using second order information, 
    and then aggregate the features using according to the correlation
    """
    bs, f_size, f_size, c_num = feature_maps.get_shape().as_list() # num of channels
    N = f_size * f_size

    ### compute the 2nd order correlation.
    feature_maps = tf.nn.relu(feature_maps)
    # feature_maps = tf.abs(feature_maps)
    x = tf.reshape(feature_maps, [bs, -1, c_num]) # (8, hw, c)

    I_hat = (-1./c_num/c_num) * tf.ones([c_num, c_num]) + (1./c_num) * tf.eye(c_num)
    I_hat = tf.tile(tf.expand_dims(I_hat, 0), [bs, 1, 1]) # # (8, c, c)
    correlation = tf.matmul(tf.matmul(x, I_hat), x, False, True) # (8, hw, hw)

    # correlation = tf.matmul(x, x, False, True) # (8, hw, hw)

    ### Aggregate features from all positions according to their correlation
    # correlation = tf.nn.softmax(correlation/(tf.sqrt(tf.cast(c_num, tf.float32))), axis = -1)
    correlation = tf.nn.softmax(correlation, axis = -1)

    aggregated_feature = tf.matmul(correlation, x) # (8, hw, c)
    aggregated_feature = tf.reshape(aggregated_feature, [bs, f_size, f_size, c_num])

    output = aggregated_feature
    # output = feature_maps + aggregated_feature
    # output = tf.concat([feature_maps, aggregated_feature], axis = -1)

    return output

def channel_second_order_att(feature_maps):
    """
    Compute the correlation between different channels using second order information, 
    and then assign attention to each channel according to its average correlation with other channels
    """
    bs, f_size, f_size, c_num = feature_maps.get_shape().as_list() # num of channels
    N = f_size * f_size

    ### compute the 2nd order correlation.
    feature_maps = tf.nn.relu(feature_maps)
    x = tf.reshape(feature_maps, [bs, -1, c_num]) # (8, hw, c)

    I_hat = (-1./N/N) * tf.ones([N, N]) + (1./N) * tf.eye(N)
    I_hat = tf.tile(tf.expand_dims(I_hat, 0), [bs, 1, 1]) # # (8, hw, hw)
    correlation = tf.matmul(tf.matmul(x, I_hat, True, False), x) # (8, c, c)

    ### Assign channels attention reweight the original feature.
    channel_att = tf.reshape(tf.reduce_mean(correlation, axis = -1), [bs, 1, 1, c_num])
    attended_feature = tf.multiply(channel_att, feature_maps)

    return attended_feature


def COAM_att(feature_maps):
    """
    Implementation of papaer "Deep Angular Embedding and Feature Correlation Attention for Breast MRI Cancer Analysis"
    """
    bs, f_size, f_size, c_num = feature_maps.get_shape().as_list() # num of channels
    N = f_size * f_size

    ### compute the GRAM correlation.
    feature_maps = tf.nn.relu(feature_maps)
    x = tf.reshape(feature_maps, [bs, -1, c_num]) # (8, hw, c)

    correlation = tf.matmul(x, x, False, True) # (8, hw, hw)

    att_map = tf.reshape(tf.reduce_mean(correlation, axis = -1), [bs, f_size, f_size, 1])

    return att_map


def self_spatial_att(feature_maps):
    """
    Firstly, the input feature maps is squeezed through global average pooling to get the global feature g.
    Then, compute the correlation between each local feature f_i and g. 
    Normalize the hw dimension correlation matrix, then generate the spatial attention through softmax activation. 
    Finally, generate the C-dim context and aggregate it with the input features.
    """
    bs, f_size, f_size, c_num = feature_maps.get_shape().as_list() # num of channels
    N = f_size * f_size

    feature_maps = tf.nn.relu(feature_maps)    
    flatten_x = tf.reshape(feature_maps, [bs, N, c_num])

    ### generate the global feature through GAP. [bs, 1, c]
    g = tf.reduce_mean(flatten_x, axis = 1, keepdims = True)

    ### compute the correlation between each local feature f_i and g
    correlation = tf.matmul(flatten_x, g, False, True) ### [bs, N, 1]                           
    spatial_weights = tf.reshape(correlation, [bs, f_size, f_size, 1])

    return spatial_weights



def channel_cor_att(feature_maps):

    bs, f_size, f_size, c_num = feature_maps.get_shape().as_list() 
    N = f_size * f_size

    feature_maps = tf.nn.relu(feature_maps)    
    flatten_x = tf.reshape(feature_maps, [bs, N, c_num])

    avg_att = tf.reduce_mean(flatten_x, axis = -1, keepdims = True) # [bs, N, 1]

    ### compute the correlation between each channel and the average attention map.
    correlation = tf.matmul(flatten_x, avg_att, True, False) ### [bs, c, 1]                           
    channel_att = tf.reshape(tf.nn.sigmoid(correlation), [bs, 1, 1, c_num])

    reweight_f = tf.reduce_mean(feature_maps * channel_att, axis = -1, keepdims = True) # [bs, h, w, 1]

    return channel_att, reweight_f    



####################################################
### Co-attention
####################################################

def co_attention_layer(f1, f2):
    """
    Use the correlation between each channel in block3 and
    attention map of block4 to assign channel attention for block3.
    Then compute the attention map of block3 as the weight sum of all channels.
    """
    with tf.variable_scope("Co_attention_layer"):
        f1 = tf.nn.relu(f1)
        f2 = tf.nn.relu(f2)

        bs, w, h, c = f2.get_shape().as_list()
        N = w * h

        f1 = tf.image.resize_images(f1, [w, h])
        original_att1 = tf.reduce_mean(f1, axis = -1, keepdims = True) ### [bs, w, h, 1]

        f1 = tf.reshape(f1, [bs, N, c]) ### [bs, N, c]

        att2 = tf.reduce_mean(f2, axis = -1, keepdims = True) ### [bs, w, h, 1]
        att2_flatten = tf.reshape(att2, [bs, N, 1])

        ### correlation between C channels in block3 and attention map of block4
        Cor = tf.matmul(f1, att2_flatten, True, False)/8.0 ### [bs, c, 1]
        ch_att = tf.nn.softmax(Cor, axis = 1)

        ### Compute the attention map of block3 as the weight sum of all channels.
        att1 = tf.matmul(f1, ch_att) ### [bs, N, 1]
        att1 = tf.reshape(att1, [bs, w, h, 1])

        # mix_att = tf.multiply(tf.multiply(original_att1, att1), att2) ### [bs, w, h, 1]
        mix_att = original_att1 + att1 + att2 ### [bs, w, h, 1]

    return att1, mix_att



def lesion_att_acc(img_index, gt_img, pixel_preds, pred_fg, prob, mode = "Net1"):
    """
    Sum of attention in the intersection area of att_mask and gt / sum of attention in the att_mask area.
    """
    print (np.max(gt_img), np.max(pred_fg), np.max(pixel_preds))
    print (gt_img.shape, pred_fg.shape, pixel_preds.shape)
    intersect = np.sum(gt_img * pred_fg) #  * pixel_preds
    total_fg = np.sum(pred_fg) # + np.sum(pixel_preds) - intersect * pixel_preds
    att_acc = round(float(intersect)/total_fg, 4)
    print (mode, "index", img_index, "Att_acc is:", att_acc, "Prob of the correct class is:", prob)

    return att_acc


def lesion_center_acc(pixel_preds, gt_img):
    fg_pos_preds = np.nonzero(pixel_preds)
    fg_center_preds = [np.mean(fg_pos_preds[0]), np.mean(fg_pos_preds[1])]
    # print ("Prediceted lesion center:", fg_center_preds)

    fg_pos_gt = np.nonzero(gt_img)
    fg_center_gt = [np.mean(fg_pos_gt[0]), np.mean(fg_pos_gt[1])]
    # print ("Ground truth lesion center:", fg_center_gt)

    # measure the distance between the prediceted lesion center and acutual lesion center.
    dist_error = round(np.sqrt(np.sum(np.square(np.array(fg_center_preds) - np.array(fg_center_gt)))), 2)
    # print ("Distance error:", dist_error)
    center_acc = np.clip(1.0/dist_error, a_min = 0.0, a_max = 0.4) * 2.5

    return dist_error, center_acc



def correlation_att(total_f_maps, bleed_feature, inflam_feature, root_path):
    """
    f_maps: [600, 8, 8, 201]
    bleed_feature, inflam_feature: [201]
    """
    bleed_feature = bleed_feature/np.linalg.norm(bleed_feature, ord = 1)
    inflam_feature = inflam_feature/np.linalg.norm(inflam_feature, ord = 1)

    for ind in range(total_f_maps.shape[0]):
        if ind >= 200:
            f_maps = total_f_maps[ind]
            f_maps = np.reshape(f_maps, (64, -1))
            fmaps_norm = np.linalg.norm(f_maps, ord = 1, axis = 1)
            bleed_correlation = np.matmul(f_maps, bleed_feature)/fmaps_norm
            inflam_correlation = np.matmul(f_maps, inflam_feature)/fmaps_norm

            image = make_pic(bleed_correlation)
            save_img(image, ind, root_path, img_name = ".jpg", mode = "heatmap")

            # bleed_prob = np.mean(bleed_correlation)
            # inflam_prob = np.mean(inflam_correlation)

            # print ("bleed:", bleed_prob, "inflam:", inflam_prob)
            # if bleed_prob > inflam_prob:
            #     print ("bleed")
            # else:
            #     print ("inflam")


def make_pic(feature):
    f_max = np.max(feature)
    f_min = np.min(feature)
    feature = (feature - f_min)/(f_max - f_min + 1e-10)
    feature = np.tile(np.reshape(feature, (8, 8, 1)), (1, 1, 3))
    image = cv2.resize(feature, (128, 128))

    return image


####################################################
### Regular operations
####################################################
def save_img(img, img_index, root_path, img_name, mode = "image"):
    img = np.uint8(255 * img)
    if mode == "image":            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif mode == "heatmap":
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img_path = os.path.join(root_path, str(img_index) + img_name)
    cv2.imwrite(img_path, img)


def generate_seg(att_map):
    """
    Convert the saliency maps of into pseudo segmentation result.
    """
    threshold = 0.6 * tf.reduce_max(att_map, axis = (1,2,3), keepdims = True)
    # threshold = tf.reduce_mean(att_map, axis = (1,2,3), keepdims = True)

    fg = tf.ones_like(att_map)
    bg = tf.zeros_like(att_map)

    seg_map = tf.where(att_map > threshold, x = fg, y = bg)

    return seg_map


def plot_scatter_correlation(att_acc, cls_prob):
    """
    Plot the scatter figure that reflect the correlation 
    between the accuracy of the att_mask and the classification prob.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Correlation between attention and classification accuracy')
    plt.xlabel('Att_acc')
    plt.ylabel('Cls_prob')

    ax1.scatter(att_acc, cls_prob, c = 'r', marker = 'o')
    plt.show()


def generate_gt_map(gt_img, dst_gt_path, img_index, mode = "Net1"):

    if gt_img.shape == (576, 576):    
        gt_img = cv2.resize(gt_img[32:544, 32:544], (128, 128))
    elif gt_img.shape == (704, 704):
        gt_img = cv2.resize(gt_img[32:672, 32:672], (128, 128))
    elif gt_img.shape == (360, 360):
        gt_img = cv2.resize(gt_img[20:340, 20:340], (128, 128))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    if mode == "Net1":
        gt_img = cv2.dilate(gt_img, kernel, iterations = 3)
    elif mode == "Net2":
        gt_img = cv2.dilate(gt_img, kernel, iterations = 3)
    gt_img = np.where(gt_img == 0, np.zeros_like(gt_img), np.ones_like(gt_img))

    cv2.imwrite(os.path.join(dst_gt_path, str(img_index) + '.jpg'), np.uint8(255 * gt_img))

    return gt_img


####################################################
### Net measure
####################################################

def get_accuracy(preds, labels):
    """
    Overall accuracy
    """
    preds = np.reshape(preds, (-1))
    labels = np.reshape(labels, (-1))
    accuracy = round(accuracy_score(y_true = labels, y_pred = preds), 5)
    """
    Per_class_recall
    """
    matrix = confusion_matrix(y_true = labels, y_pred = preds)
    print ("confusion_matrix:", matrix)
    recalls = matrix.diagonal().astype('float')/matrix.sum(axis = 1)

    normal_recall = round(recalls[0], 5)
    bleed_recall = round(recalls[1], 5)
    inflam_recall = round(recalls[2], 5)

    """
    Cohen kappa
    """  
    kappa = round(cohen_kappa_score(y1 = preds, y2 = labels), 5)

    # print ("kappa", kappa)
    # print normal_recall, bleed_recall, inflam_recall, kappa

    return accuracy, normal_recall, bleed_recall, inflam_recall, kappa


def print_scores(normal_recall, bleed_recall, inflam_recall, kappa):

    print ("Normal recall:", normal_recall)
    print ("Bleeding recall:", bleed_recall)
    print ("Inflam recall:", inflam_recall)
    print ("Cohen kappa score:", kappa)    


def superior_net2(prob1, prob2):
    """Compute the ratio of prob2 larger than prob1."""
    total_num = len(prob1)
    print ("Total number:", total_num)

    better2_index = np.nonzero(np.maximum(0, prob2 - prob1))
    # print ("Images with more accurate predictions from net2:", better2_index)

    better2_num = np.count_nonzero(np.maximum(0, prob2 - prob1))
    print ("The number of images have larger prob2 than prob1:", better2_num)
    ratio = float(better2_num)/float(total_num)

    return ratio
