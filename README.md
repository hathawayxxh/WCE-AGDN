# Zoom in Lesions for Better Diagnosis: Attention Guided Deformation Network for WCE Image Classification

by [Xiaohan Xing](https://sites.google.com/view/xhxing), [Yixuan Yuan](http://www.cityu.edu.hk/stfprofile/yixuyuan.htm), [Max Q.-H. Meng](https://www.ee.cuhk.edu.hk/~qhmeng/).

This paper is accepted by Transactions on Medical Imaging (TMI) 2020. 

## Introduction:
In order to make better diagnosis of the small lesions in WCE images, we propose a two-branch Attention Guided Deformation Network (AGDN) which utilizes attention maps to localize and zoom in lesion regions, thus enabling better inspection. In order to improve the feature representation and discrimination ability of the AGDN model, we devise and insert a novel Third-order Long-range Feature Aggregation (TLFA) module into the network. What's more, we propose a novel Deformation based Attention Consistency (DAC) loss to achieve mutual promotion of the two branches. 


## Framework
![image](https://github.com/hathawayxxh/WCE-AGDN/blob/master/overview_new1.png)

## Requirement:
Tensorflow 1.8
Python 2.7

## Code Base Structure:
- run_2dense_net.py:  main function, train or test the model.
- models.
  - AGDN_model.py:   implementation of the AGDN model used in the paper. 

## Comparison between the TLFA module and nonlocal module:
AGDN_nonlocal.py

ref: [Non-Local Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_NonLocal_Neural_Networks_CVPR_2018_paper.pdf)


