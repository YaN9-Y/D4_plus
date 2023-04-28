D4+: Robust Unpaired Image Dehazing via Density and Depth Decomposition
===============================================
This is the PyTorch implementation of the paper 'Robust Unpaired Image Dehazing via Density and Depth Decomposition'.

Introduction
---------------------------------
This paper presents an advanced version of our original work, termed D4+. The upgraded framework exploits a dual contrastive perceptual loss to further improve the performance of both haze removal and generation. Moreover, it successfully extends the scope of training from indoor synthetic datasets to real outdoor scenes and achieves remarkable improvement in dehazing performance. Such extension together with the introduced contrastive loss makes the whole framework more robust and effective for real-world hazy scenes. More detailed experiments on both synthetic and real datasets are conducted to validate the effectiveness of our method.

Prerequisites
---------------------------------
* Python 3.7
* Pytorch 1.7.1
* NVIDIA GPU + CUDA cuDNN

Datasets
---------------------------------
### 1.Data for testing
After downloading the dataset, please use scripts/flist.py to generate the file lists. For example, to generate the training set file list on the SOTS-indoor testset, you should run:

```
python scripts/flist.py --path path_to_SOTS_indoor_hazy_path --output ./datasets/sots_test_hazy_indoor.flist
```

Please notice that the ground truth images of SOTS-indoor have additional white border, you can crop it first.

### 2.Data for training (Still organizing)
For training on the synthetic indoor dataset, we used [ITS](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) dataset, you can follow the operations above to generate the training file lists.

```
python scripts/flist.py --path ITS_train_hazy_path --output ./datasets/its_train_hazy.flist
python scripts/flist.py --path ITS_train_gt_path --output ./datasets/its_train_gt.flist
```

For training on real outdoor scenes, we used reside-unpaired from [RefineDNet](https://github.com/xiaofeng94/RefineDNet-for-dehazing) and generate sky mask basing on it. The datasets together with our generated mask is available at [Download link](https://drive.google.com/file/d/1pfXZCFZ-8F53LdHyXHuiQdFNFbQ0qBmV/view?usp=share_link). Please notice that some masks for hazy images may be inaccurate due to dense haze. But these masks work fine to eliminate the ambiguity of sky. However, this mask dataset is not appropriate for training segmentation models. 


Getting Started
--------------------------------------
To use the pre-trained models, download it from the following link then copy it to the corresponding checkpoints folder, like `./checkpoints/quick_test`

[Pretrained model on ITS](https://drive.google.com/file/d/1_JA3UHVpBym4wARDM8GkcsCgEI6x2irP/view?usp=share_link) | [Pretrained model on Real](https://drive.google.com/file/d/1hshyzMCXYrPHUzwZk2rji9ExEQDKbOD_/view?usp=share_link)


### 1.Training (Still Organizing)
1)Prepare the training datasets following the operations in the Dataset part.
2)Add a config file 'config.yml' in your checkpoints folder. We provide an example checkpoints folder and config file in `./checkpoints/train_example`, remember to fill the correct dataset in the config file. 
3)Train the model, for example:

```
python train.py --model 1 --checkpoints ./checkpoints/train_example
```

### 2. Testing
1)Prepare the testing datasets following the operations in the Dataset part.
2)Put the trained weight in the checkpoint folder 
3)Add a config file 'config.yml' in your checkpoints folder. We have provided example checkpoints folder and config files in `./checkpoints/`, 
4)Test the model, for example:
```
python test.py --model 1 --checkpoints ./checkpoints/test_sots_indoor_example
```
For quick testing, you can download the checkpoint and put it to the corresponding folder `./checkpoints/test_rtts_example` or `./checkpoints/test_sots_indoor_example` and run test on our example images directly using

```
python test.py --model 1 --checkpoints ./checkpoints/test_sots_indoor_example  OR ./checkpoints/test_rtts_example
```

