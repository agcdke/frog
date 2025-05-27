# AgriBlazeU-Net
AgriBlazeU-Net is an U-Net based semantic segmentation model that incorporates a custom CNN attention module. This module replaces the spatial attention module of Convolutional Block Attention Module (CBAM) by MobileViT block to leverage self-attention and to enhance the attention capability. The training is performed from scratch using multi-gpu(2 GPUs) strategy on two publicly available RGB datasets, i.e., CWFID or Crop/Weed Field Image dataset(Haug et al.), and Crop vs weed discrimination dataset(Bosilj et al.). CWFID usecase is used in this repository.

## Getting Started
### Annotation
To infer labeled pixels automatically during model training, *data_dir/cwfid_class_dict.csv* file presents labeled information, i.e., weed is red (RGB value: 255,0,0), crop is green (RGB value: 0,255,0), and soil as unlabeled is black (RGB value: 0,0,0). The images belongs to train and test datasets are referred at *data_dir/cwfid_train_test_split.yaml* file, where train imagaes are spilt to create train and validation datasets. The image augmentation is applied only on train dataset. 

### Installation
Please create conda environment following *"coming_soon"* file. Here, TensorFlow 2.15 is considered for the exepriments.

### Slurm
To perform experiment, please run slurm job

```shell
# Run slurm job
$ sbatch cwfid.slurm
```
The experiment result is stored at *SBATCH --output* file (mentioned at cwfid.slurm)

### Model Selection

The two variables *model_arch_module* and *model_arch_type* are used for model selection at *src/mgpu_unet_cwfid.py* file.

| Model Name             | Arch Module          | Arch Type                        |
| :--------------------- | :------------------- | :------------------------------- |
| BlazeFace              | imgcls_blaze         | blz                              |
| BlazeFace with CBAM    | imgcls_blaze_cbam    | bcbam                            |
| BlazeFace with CBwSSAM | imgcls_blaze_cbwssam | T1H8B8 (Type-1), T2H8B8 (Type-2) |

The module **T1H8B8** (i.e., *mvit_h8b8*) denotes *(a) Type-1 CBwSSAM* module, where *h8b8* refers 8 head and 8 block in the Transformers architecture. Similarly, module **T2H8B8** (i.e., *cbmvit_h8b8*) denotes *(b) Type-2 CBwSSAM* module, where *h8b8* refers 8 head and 8 block in the Transformers architecture. The [module architecture](assets/img/CBwSSAM.png) is presented.

### Loss Function
The **Unet3pHybridLoss** loss function(S. Jadon et al.) is used during model training. 

### Evaluation
A dataset is split over train, val, and test datasets. The image augmentation is performed only on train dataset. 
The final evaluation is done on test dataset using *src/eval_unet_cwfid.py* file.