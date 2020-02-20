# MEAL: Multi-Model Ensemble via Adversarial Learning

This is the official `PyTorch` implementation for paper:

**[MEAL: Multi-Model Ensemble via Adversarial Learning](https://arxiv.org/abs/1812.02425)** (AAAI 2019, **Oral**).

[Zhiqiang Shen](http://zhiqiangshen.com)\*, [Zhankui He](https://aaronheee.github.io/)\*, [Xiangyang Xue](https://scholar.google.com/citations?user=DTbhX6oAAAAJ&hl=en).

The key idea of this work is distilling diverse knowledge from different trained models (teachers) into a single student network, in order to *learn an ensemble of multiple models without incurring additional testing costs*. We use adversarial-based learning strategy where we define a block-wise training loss to guide and optimize the predefined student network to recover the knowledge in teacher models, and to promote the discriminator network to distinguish teacher vs. student features simultaneously.

The student and teacher networks we implemented are listed in `\models`, and it is also easy to add new networks in our repo. The corresponding author of this paper is: [Dr. Zhiqiang Shen](http://zhiqiangshen.com).

If you find this helps your research, please cite:

	@inproceedings{shen2019MEAL,
		title = {MEAL: Multi-Model Ensemble via Adversarial Learning},
		author = {Shen, Zhiqiang and He, Zhankui and Xue, Xiangyang},
		booktitle = {AAAI},
		year = {2019}
	}

## Quick Start
- git clone this repo
- download pre-trained teachers (on CIFAR-10):
```bash
sh ./scripts/download_pretrained_models.sh
``` 
(You can also manually download them [here](https://drive.google.com/open?id=12bVZFoFFAYYTGGXzMBDnnfwD1lJBxZY9).)
- for single MEAL like `teacher: vgg, student: vgg`:
```bash
python main.py --gpu_id 0 --teachers [\'vgg19_BN\'] --student vgg19_BN --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --out_layer [0,1,2,3,4] --out_dims [10000,5000,1000,500,10] --gamma [0.001,0.01,0.05,0.1,1] --eta [1,1,1,1,1] --name vgg_test
```
- for ensemble MEAL like `teachers: vgg19, densenet, dpn92,resnet18, preactresnet18; student:densenet`:
```bash
python main.py --gpu_id 0 --lr 0.1 --batch_size 256 --teachers [\'vgg19_BN\',\'dpn92\',\'resnet18\',\'preactresnet18\',\'densenet_cifar\'] --student densenet_cifar --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name 5_ensemble_for_densenet --out_layer [-1] 
```

## Environment
Python 3.6+

PyTorch 0.40+

Numpy 1.12+ 

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,a*150)`
- `0.01` for epoch `[a*150,a*250)`
- `0.001` for epoch `[a*250,a*350)`

The factor `a` varies with number of teacher networks, between 1 and 2.

## ImageNet model
Our trained ResNet-50 (**the accuracy is even comparable to PyTorch official [ResNet-152](https://pytorch.org/docs/stable/torchvision/models.html)**):

|Models| Top-1 error (%) | Top-5 error (%)  | URL
|:-------|:-----:|:-----:|:-----:|
|ResNet-50 | 23.85 | 7.13| - |
|ResNet-101 | 22.63 | 6.44| - |
|ResNet-152 | 21.69 | 5.94| - |
|**Our ResNet-50**| **21.79** | **5.99**| [Download (102.5M)](https://drive.google.com/open?id=1x6SUiPWbqIKtdF_XRtEBQuinRfHUiRvm) |
