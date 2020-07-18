# Zoom in Lesions for Better Diagnosis: Attention Guided Deformation Network for WCE Image Classification \[TMI 2020]

by [Xiaohan Xing](https://sites.google.com/view/xhxing), [Yixuan Yuan](http://www.cityu.edu.hk/stfprofile/yixuyuan.htm), [Max Q.-H. Meng](https://www.ee.cuhk.edu.hk/~qhmeng/).

This paper is accepted by Transactions on Medical Imaging (TMI) 2020. 

## Introduction
In order to make better diagnosis of the small lesions in WCE images, we propose a two-branch Attention Guided Deformation Network (AGDN) which utilizes attention maps to localize and zoom in lesion regions, thus enabling better inspection. In order to improve the feature representation and discrimination ability of the AGDN model, we devise and insert a novel Third-order Long-range Feature Aggregation (TLFA) module into the network. What's more, we propose a novel Deformation based Attention Consistency (DAC) loss to achieve mutual promotion of the two branches. 


## Framework
![image](https://github.com/hathawayxxh/WCE-AGDN/blob/master/overview_new1.png)

## Requirement
Tensorflow 1.8
Python 2.7

## Code Base Structure
- **run_2dense_net.py:** main function, train or test the model.
- **models.**
  - **AGDN_model.py:** implementation of the AGDN model used in the paper. 
  - **AGDN_SONA.py:**    implementation of the "B2 + 3TLFA-" model in the paper, replace the deformale convolution in TLFA modules with pointwise convolution,
                          the new module is called TLFA- (in the paper) or second-order non-local (SONA) modules (in the code).
  - **AGDN_nonlocal.py:**   implementation of the "B2 + 3Nonlocal" model in the paper, replace the 3 TLFA modules in the AGDN model with nonlocal modules.
  - **AGDN_crop_input2.py:**   implementation of the "B1 + cropping" model in the paper, using the cropped image rather than the deformed image as the input of branch2.
  - **AGDN_DML.py:**    replace the DAC loss in the AGDN model with deep mutual learning (DML) loss, which constrain the logits predictions of the two branches to be similar      (not contained in the paper, only appear in the response letter).
  - **AGDN_cam_att.py:**    replace the average attention maps with the class activation maps (CAM) to provide guidance for the image deformation (not contained in the paper, only appear in the response letter).
  - **AGDN_gradcam.py:**    replace the average attention maps with the grad-cam to provide guidance for the image deformation (not contained in the paper, only appear in the response letter).
- **high_low_res_data_provider.py:**  Provide the images both in low and high resolution, the low res images are used as input1, attention map1 is used to sample the high res images to provide input2. 
- **trainable_image_sampler.py:**    perform attention guided image deformation (AGD) using the combination of: customized grids generator (CGG) and structured grids generator (SGG).
- **ops.py & ops1.py:**    contain functions that are utilized in the model.
- **measure_att_map.py:**    evaluate the attention maps by precision-recall curves, F-measure, Mean absolute error.


## Train and Test
Here are example commands for training or testing the AGDN model.
- **Train:** 

python run_2dense_net.py --train
- **Test:** 

python run_2dense_net.py --test

## Citation
If you find our work useful in your research or if you use parts of this code, please consider citing our paper:

to be uploaded.

## Questions
Please contact "<xhxing@link.cuhk.edu.hk>".

