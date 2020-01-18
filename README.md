# WCE-AGDN
Zoom in Lesions for Better Diagnosis: Attention Guided Deformation Network for WCE Image Classification

by [Xiaohan Xing](https://sites.google.com/view/xhxing), [Yixuan Yuan](http://www.cityu.edu.hk/stfprofile/yixuyuan.htm), [Max Q.-H. Meng](https://www.ee.cuhk.edu.hk/~qhmeng/)

## Introduction:
In order to make better diagnosis of the small lesions in WCE images, we propose a two-branch Attention Guided Deformation Network (AGDN) which utilizes attention maps to localize and zoom in lesion regions, thus enabling better inspection. In order to improve the feature representation and discrimination ability of the AGDN model, we devise and insert a novel Third-order Long-range Feature Aggregation (TLFA) module into the network. What's more, we propose a novel Deformation based Attention Consistency (DAC) loss to achieve mutual promotion of the two branches. 


## Framework
![image](https://github.com/hathawayxxh/WCE-AGDN/blob/master/overview_new1.png)
Given a high-resolution WCE image I_H, it is first resized to $I_L$ with a resolution 128 $\times$ 128 and then fed into branch1, which is constructed by a DenseNet equipped with TLFA modules. 
Attention map $A(I_L)$ indicating the lesion locations is obtained by taking the channel-wise average of feature maps in the block4 of branch1. 
Then, we utilize the AGD module to transform $I_H$ into $I_{DL}$ with a resolution 128 $\times$ 128. Compared with $I_L$, lesion areas in $I_{DL}$ are amplified while irrelevant background features are compressed. 
Taking $I_{DL}$ as the input, branch2 generates image-level prediction2 and attention map $A(I_{DL})$. 
% Note that the two branches share identical architecture but with independent parameters.
Finally, the squeezed global features in these two branches are concatenated to make the final classification prediction. 
The network is trained by a joint loss function containing cross-entropy losses ($L_{C1}$, $L_{C2}$, $L_{C3}$) and DAC losses ($L_{DAC1}$, $L_{DAC2}$).
