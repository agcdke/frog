# AgriBlazeNet
AgriBlazeNet is an image classification model that incorporates BlazeFace feature extraction network. A a custom CNN attention module with Mobile-Vision Transformers block is introduced, where the spatial attention module of Convolutional Block Attention Module (CBAM) is replaced by MobileViT block to leverage self-attention and to enhance the attention capability. The training is performed from scratch using multi-gpu(2 GPUs) strategy on three publicly available RGB datasets, i.e., DeepWeeds(Olsen et al.), Plant Seedlings(Giselsson et al.), and Plant Stress(S. Ghosal et al.) datasets. Plant Stress usecase is used in this repository. For model architecture, please refer to *assets/img/AgriBlazeNet.png* file.

## Getting Started
### Installation
Please create conda environment following *"coming_soon"* file. Here, TensorFlow 2.15 is considered for the exepriments.

### Slurm
To perform experiment, please run slurm job

```shell
# Run slurm job
$ sbatch soybeanstress.slurm
```
The experiment result is stored at *SBATCH --output* file (mentioned at soybeanstress.slurm)

### Model Selection

The two variables *model_arch_module* and *model_arch_type* are used for model selection at *src/multigpu_imgcls_soybeanplantstress.py* file.

| Model Name             | Arch Module          | Arch Type                 |
| :--------------------- | :------------------- | :------------------------ |
| BlazeFace              | imgcls_blaze         | blz                       |
| BlazeFace with CBAM    | imgcls_blaze_cbam    | bcbam                     |
| BlazeFace with CBwSSAM | imgcls_blaze_cbwssam | mvit_h4b4, cbmvit_h4b4    |

The module **mvit_h4b4** denotes *(a) Type-1 CBwSSAM* module, where *h4b4* refers 4 head and 4 block in the Transformers architecture. Similarly, module **cbmvit_h4b4** denotes *(b) Type-2 CBwSSAM* module, where *h4b4* refers 4 head and 4 block in the Transformers architecture. The [module architecture](assets/img/CBwSSAM.png) is presented.

### Evaluation and Model Quantization
A dataset is split over train, validation, and test datasets. The image augmentation is performed only on train dataset. The final evaluation is done on test dataset using *src/eval_imgcls_soybeanplantstress.py* file. Post-training model quantization is done using LiteRT(former TensorFlow Lite).

## Acknowledgement:
* Image classification evaluation metrics: [wandb](https://wandb.ai/site/) 