--- AGDN_model.py   
       implementation of the AGDN model used in the paper. 
--- AGDN_SONA.py    
       implementation of the "B2 + 3TLFA-" model in the paper, replace the deformale convolution in TLFA modules with pointwise convolution,
       the new module is called TLFA- (in the paper) or second-order non-local (SONA) modules (in the code).
--- AGDN_nonlocal.py   
       implementation of the "B2 + 3Nonlocal" model in the paper, replace the 3 TLFA modules in the AGDN model with nonlocal modules.
--- AGDN_crop_input2.py   
       implementation of the "B1 + cropping" model in the paper, using the cropped image rather than the deformed image as the input of branch2.
--- AGDN_DML.py    
       replace the DAC loss in the AGDN model with deep mutual learning (DML) loss, which constrain the logits predictions of the two branches to be similar (not contained in the paper, only appear in the response letter).
--- AGDN_cam_att.py    
       replace the average attention maps with the class activation maps (CAM) to provide guidance for the image deformation (not contained in the paper, only appear in the response letter).
--- AGDN_gradcam.py    
       replace the average attention maps with the grad-cam to provide guidance for the image deformation (not contained in the paper, only appear in the response letter).
