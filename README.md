![Alt text](assets/img/frog-logo.png=250x250)

# FROG
FROG (FROm Ground) is an initiative to offer several vision and language services for agricultural (business) applications. The objective of this initiative is to increase the sustainability of food supply from agricultural farms.


## Getting Started
### Vision Module
Vision module currently offers two services, i.e., image classification, and U-Net based semantic segmentation. The services are available at *vision* directory. 

**AgriBlazeNet:** It offers four variants of an image classification [model](assets/img/AgriBlazeNet.png), which incorporates BlazeFace feature extraction network as a baseline. It incorporates CBAM module(S. Woo et al.) in one variant. It also incorporates a custom CNN attention module known as Convolutional Block with Spatial Self-Attention Module(CBwSSAM), where the spatial attention of CBAM is replaced by the Vision Transformer (particularly MobileViT) to enhance the attention capability for several agricultural applications. The architecture of two variants of [CBwSSAM](assets/img/CBwSSAM.png) is presented. 

**AgriBlazeU-Net** It offers an U-Net based semantic segmentation model, where CBwSSAM variants are incorporated. The annotated RGB images are inferred automatically during model training. For example, weed, crop and soil are annotated as red, green and black respectively. The annotated information is saved in *unet/data_dir/cwfid_class_dict.csv* file for CWFID dataset, and inferred automatically during model training. The information about train and testset is mentioned at *unet/data_dir/cwfid_train_test_split.yaml* file.

### Language Module
Language module currently offers two services, i.e., Question-Answering, and Chatbot. The services are available at *language* directory. 

**Question-Answering** It offers a preliminary Retrieval-Augmented Generation(RAG) based Question-Answering framework which uses LangChain, Chroma, and Ollama. It is available at *language/agriqa* directory.

**Chatbot** It offers a preliminary Retrieval-Augmented Generation(RAG) based Chatbot framework which uses LangGraph, Chroma, and Ollama. It is available at *language/agrichatbot* directory.

## Acknowledgement:
Verify markdown texts at [Markdown Live Preview](https://markdownlivepreview.com)