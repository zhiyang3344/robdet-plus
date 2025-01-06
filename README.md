# Improving Robust Out-of-distribution Detection via Effective Self-labeling in Adversarial Training

This code is implemented by PyTorch, and we have tested the code under the following environment settings:\
python = 3.8.5\
numpy == 1.19.2\
torch = 1.7.1\
torchvision = 0.8.2

#### 1. Install our modified attacks (modified from Auto-Attack):  
cd autoattack\
python setup.py build\
pip install -e .


#### 2. Download auxliiary training OOD datasets and test OOD datasets
Please refer to https://github.com/hendrycks/outlier-exposure and https://github.com/jfc43/informative-outlier-mining

#### 3. Training and Evaluation
##### Train Detections with 'other' class (SOFL, ATOM, RobDet, and RobDet+):
python train_obranch.py --model_name wrn-40-4 --dataset cifar10 --num_out_classes --training_method pair-mixpgd-ce --ood_training_method clean-mixpgd-ce_out --oadv_multi_stage --gpuid $GPU-ID$ --tiny_file $TINY_FILE$ --model_dir $MODEL_DIR$

##### Train Detections without OOD classes (AT, ClnAT, ACET, RCE):
python train_unif.py --model_name wrn-40-4 --dataset cifar10 --training_method pgd-ce --ood_training_method pgd-ce --gpuid $GPU-ID$ --tiny_file $TINY_FILE$ --model_dir $MODEL_DIR$


##### Evaluate Detections with OOD class:
python ood_eval_out_branch.py --model_name wrn-40-4 --dataset cifar10 --num_out_classes $V$ --gpuid $GPU-ID$ --model_file $MODEL_FILE$

##### Evaluate Detections without OOD class:
python ood_unif.py --model_name wrn-40-4 --dataset cifar10 --gpuid $GPU-ID$ --model_file $MODEL_FILE$


#### 4. References:
[1] ODIN: https://github.com/facebookresearch/odin  
[2] OE: https://github.com/hendrycks/outlier-exposure  
[3] ATOM: https://github.com/jfc43/informative-outlier-mining    
[4] Auto-Attack: https://github.com/fra31/auto-attack  

