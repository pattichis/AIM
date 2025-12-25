# AIM
This document contains links to select datasets, models, papers, and related PyTorch links related to AI in Medical Image Analysis.<br />
The links have been verified in December, 2025. 

🔴 **Important:** Click on the Outline button (upper-right button in GitHub) for a table of contents and to jump to a particular topic.<br />
🔴 **Important:** Right-click on each link to open in a new browser window.

Please reference:

A.S. Panayides *et al.*, “Artificial Intelligence in Medical Image Analysis: Advances, Clinical Translation, and Emerging Frontiers,” submitted to
*IEEE Journal of Biomedical and Health Informatics*, 2025.

```bibtex
@article{AIinMedicalImaging,
  title={Artificial Intelligence in Medical Image Analysis: Advances, Clinical Translation, and Emerging Frontiers},
  author={Panayides, A. S., and Chen, H. and Filipovic, N. D. and Geroski, T. and Hou, K. and Lekadir, K. and 
  Marias, K. and Matsopoulos, G. and Papanastasiou, G. and Sarder, P. and Tourassi, G. and  
  Tsaftaris, S. A. and Amini, A. and Fu, H. and Kyriacou, E. and Loizou, C. P. and Zervakis, M. and 
  Saltz, J. H. and Shamout, F. E. and Wong, K. C. L. and Yao, J. and Fotiadis, D. I. and
  Pattichis, C. S. and Pattichis, M. S.}
  journal={IEEE Journal on Biomedical and Health Informatics},
  year={2026}
}
```

For updates, email Prof. Marios S. Pattichis at [pattichi@unm.edu](mailto:pattichi@unm.edu).

# Open Models for Digital Image Analysis

# A generalist vision–language foundation model for diverse biomedical tasks
* [BiomedGPT is pre-trained and fine-tuned with multi-modal & multi-task biomedical datasets](https://github.com/taokz/BiomedGPT)
* [BiomedGPT paper link](https://www.nature.com/articles/s41591-024-03185-2)
* [BiomedGPT Google Drive based Google Colab](https://colab.research.google.com/drive/1AMG-OwmDpnu24a9ZvCNvZi3BZwb3nSfS?usp=sharing#scrollTo=nEQJH5MoT-we)

## PyTorch Image encoders/backbones
* [pytorch-image-models: The largest collection of PyTorch image encoders/backbones. Including train, eval, inference, export scripts, and pretrained weights](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#getting-started-documentation)

## Vision Transformer Implementations
* [vit-pytorch: PyTorch-based implementations of vision transformer architectures](https://github.com/lucidrains/vit-pytorch/blob/main/README.md)
* [vit-tensorflow: Tensorflow-based implementations of vision transformer architectures](https://github.com/taki0112/vit-tensorflow)
* [pytorch-image-models: Contains pre-trained trasnformer models](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#getting-started-documentation)
 

## Python libraries for Pathology image analysis
* [HistoQC is an open-source quality control tool for digital pathology slides](HTTPS://GITHUB.COM/CHOOSEHAPPY/HISTOQC)
* [CLAM: A Deep-Learning-based Pipeline for Data Efficient and Weakly Supervised Whole-Slide-level Analysis](https://github.com/MAHMOODLAB/CLAM)
* [HistomicsTK is a Python package for the analysis of digital pathology images](HTTPS://GITHUB.COM/DIGITALSLIDEARCHIVE/HISTOMICSTK)
* [Slideflow is a deep learning library for digital pathology, offering a user-friendly interface for model development](HTTPS://GITHUB.COM/SLIDEFLOW/SLIDEFLOW?TAB=README-OV-FILE) 
* [Sarder Lab: Codes for computational pathology from Pinaki Sarder's lab](https://github.com/SarderLab)
* [MoPaDi - Morphing Histopathology Diffusion](HTTPS://GITHUB.COM/KATHERLAB/MOPADI)

# Cancer imaging
* [Low-Rank Adaptation of Pre-Trained Large Vision Models for Improved Lung Nodule Malignancy Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10843806), [NLSTx Dataset: A subset of difficult lung nodules from the NLST database](https://github.com/benVZ/NLSTx).
* [NCI's CPTAC: Clinical Proteomic Tumor Analysis Consortium (proteomics, genomics, histopathology)](HTTPS://GDC.CANCER.GOV/ABOUT-GDC/CONTRIBUTED-GENOMIC-DATA-CANCER-RESEARCH/CLINICAL-PROTEOMIC-TUMOR-ANALYSIS-CONSORTIUM-CPTAC) 
* [NCI Imaging Data Commons (IDC) is a cloud-based repository of publicly available cancer imaging data co-located with analysis and exploration tools](HTTPS://DATACOMMONS.CANCER.GOV/REPOSITORY/IMAGING-DATA-COMMONS) 
* [HTAN is a National Cancer Institute (NCI)-funded Cancer MoonshotSM initiative to construct 3-dimensional atlases of the dynamic cellular, morphological, and molecular features of human cancers as they evolve from precancerous lesions to advanced disease](https://humantumoratlas.org/)

# Open datasets focused on pathology
* [The KPMP is a multi-year collaboration of leading research institutions to study patients with kidney disease](
https://www.kpmp.org/about-kpmp)
* [HUBMAP: Human BioMolecular Atlas Program Data Portal: An open platform to discover, visualize and download standardized healthy single-cell and spatial tissue data](HTTPS://PORTAL.HUBMAPCONSORTIUM.ORG)
* [TCGA: The Cancer Genome Atlas Program](HTTPS://WWW.CANCER.GOV/CCG/RESEARCH/GENOME-SEQUENCING/TCGA)
* [GTEx: The Genotype-Tissue Expression (GTEx) Portal is a comprehensive public resource for researchers studying tissue and cell-specific gene expression and regulation across individuals, development, and species, with data from 3 NIH projects](HTTPS://GTEXPORTAL.ORG/HOME/)
* [Download and visually explore data to understand the functionality of human tissues at the cellular level with Chan Zuckerberg CELL by GENE Discover (CZ CELLxGENE Discover](https://cellxgene.cziscience.com/)
* [The Human Cell Atlas is a global consortium that is mapping every cell type in the human body, creating a 3-dimensional Atlas of human cells to transform our understanding of biology and disease](https://www.humancellatlas.org) 
* [BRIDGE2AI's Functional genomics project (protein-protein interactions, single-cell imaging (Trey Ideker at UCSD is PI)](HTTPS://BRIDGE2AI.ORG/CM4AI_/)
* [4 dimensional nucleome (imaging and omics to relate the spatial orientation of the nucleome to gene regulation)](https://4dnucleome.org/)
* [Apollo: The Applied Proteogenomics OrganizationaL Learning and Outcomes (APOLLO) Network (DOD, VA, NCI; cancer program with genomics, proteomics, pathology and excellent longitudinal clinical data of veterans](https://www.cancer.gov/about-nci/organization/cbiit/projects/apollo)
* [CELLxGENE is a suite of tools that help scientists to find, download, explore, analyze, annotate, and publish single cell datasets](https://cellxgene.cziscience.com/docs/01__CellxGene)
* [GDS: This database stores curated gene expression DataSets, as well as original Series and Platform records in the Gene Expression Omnibus (GEO) repository](https://www.ncbi.nlm.nih.gov/gds/?term)

# Echocardiography
## [Echonet datasets and models](https://github.com/echonet)
* [EchonNet-LVH: A Large Parasternal Long Axis Echocardiography Video Dataset, Model, and  Paper](https://echonet.github.io/lvh/), [model](https://github.com/echonet/lvh), [paper](https://jamanetwork.com/journals/jamacardiology/fullarticle/2789370).
* [EchoNet-Pediatric: A Large Pediatric Echocardiography Video Dataset and Model link](https://echonet.github.io/pediatric/index.html), [paper](https://www.clinicalkey.com/#!/content/playContent/1-s2.0-S0894731723000688?returnurl=https:%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0894731723000688%3Fshowall%3Dtrue&referrer=https:%2F%2Fpubmed.ncbi.nlm.nih.gov%2F).
* [EchoNet-Dynamic: Interpretable AI for beat-to-beat cardiac function assessment Dataset, Model, and Paper](https://github.com/echonet/dynamic)
* [EchoNet: Tee-View-Classifier datasets and paper](https://aimi.stanford.edu/datasets/echonet-tee-view-classifier) and [model](https://github.com/echonet/tee-view-classifier)
* [EchoNet-Synthetic: Privacy-preserving Video Generation for Safe Medical Data Sharing](https://github.com/HReynaud/EchoNet-Synthetic) and [paper](https://arxiv.org/abs/2406.00808) (also see Generative AI Video Models).

# Foundation Models

## Foundation model - related general libraries
* [Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)](HTTPS://GITHUB.COM/FACEBOOKRESEARCH/MASK2FORMER)
* [Detectron2 is Facebook AI Research's next-generation library that provides state-of-the-art detection and segmentation](HTTPS://GITHUB.COM/FACEBOOKRESEARCH/DETECTRON2)

## Foundation models for pathology image analysis
*  [CHIEF - Clinical Histopathology Imaging Evaluation Foundation Model (focused on cancer)](https://github.com/HMS-DBMI/CHIEF)
* [UNI HIPT: Towards a general-purpose foundation model for computational pathology](HTTPS://DOI.ORG/10.1038/S41591-024-02857-3)
* [CellViT++: Energy-Efficient and Adaptive Cell Segmentation and Classification Using Foundation CELLVIT](HTTPS://GITHUB.COM/TIO-IKIM/CELLVIT-PLUS-PLUS)
* [Cellpose-SAM: cell and nucleus segmentation with superhuman generalization](https://github.com/MOUSELAND/CELLPOSE)   
* [Prov-GigaPath A whole-slide foundation model for digital pathology from real-world data](https://github.com/prov-gigapath/prov-gigapath)
* [H0-mini is a lightweight foundation model for histology](https://huggingface.co/bioptimus/H0-mini)
* [UNI: Towards a General-Purpose Foundation Model for Computational Pathology](HTTPS://GITHUB.COM/MAHMOODLAB/UNI)

## Vision language foundation models for pathology
* [CONCH: A Vision-Language Foundation Model for Computational Pathology](https://github.com/mahmoodlab/CONCH)

## [Foundation Model for Endoscopy Video Analysis](https://github.com/openmedlab/Endo-FM)
* Contains links to 10 different endoscopy video datasets.
* A large-scale endoscopic video dataset with over 33K video clips.
* Supports 3 types of downstream tasks, including classification, segmentation, and detection.

## SAM foundation models for image and video segmentation, and 3D reconstruction
* [SAM (Segment Anything Model, META, 2023)](HTTPS://GITHUB.COM/FACEBOOKRESEARCH/SEGMENT-ANYTHING)
* [SAM2 foundation model for video](https://github.com/facebookresearch/sam2)
* [SAM2 paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
* [SAM3 A unified model for detection, segmentation, and tracking of objects in images and video using text, exemplar, and visual prompts](https://ai.meta.com/blog/segment-anything-model-3/)
* [SAM3 paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
* [SAM 3D contains two state-of-the-art models that enable 3D reconstruction of objects and humans from a single image.](https://ai.meta.com/sam3d/), [GitHub](https://github.com/facebookresearch/sam-3d-objects).

# Instructional Medical Videos
## A dataset for medical instructional video classification and question answering
* [MedVidQACL datasets and models](https://github.com/deepaknlp/MedVidQACL)
* [Paper](https://www.nature.com/articles/s41597-023-02036-y)
  
## [How Well Can General Vision-Language Models Learn Medicine By Watching Public Educational Videos?](https://arxiv.org/abs/2504.14391)
* Main website with model: [OpenBiomedVid](https://github.com/zou-group/OpenBiomedVid)
* [OpenBiomedVid dataset](https://huggingface.co/datasets/connectthapa84/OpenBiomedVid)
* [SurgeryVideoQA](https://huggingface.co/datasets/connectthapa84/SurgeryVideoQA)
* [MIMIC-IV-ECHO: Echocardiogram Matched Subset](https://physionet.org/content/mimic-iv-echo/0.1/)
* [Related OpenAI o3 and o4-mini System](https://openai.com/index/o3-o4-mini-system-card/)
* [OpenAI models](https://github.com/openai/gpt-oss)

# Generative AI Image Models 
* [Deterministic Medical Image Translation via High-fidelity Brownian Bridges](https://arxiv.org/pdf/2503.22531) 
  (CVPR 2025 (preprint) paper only).
  General (cross-modality translation) using MRI/CT simulated datasets (no fixed subject count). This is an 
  image-to-image method. Deterministic diffusion using Brownian bridge paths to connect source and target modalities,
  improving realism and consistency without stochastic sampling.
* [GDM-VE: Geodesic Diffusion Models for Medical Image-to-Image Generation (2025)](https://github.com/mirthAI/GDM-VE) (GitHub link, paper link also).
  MRI & CT (brain, thoracic; open datasets). Geodesic Diffusion Model. Image-to-image method. Introduces a geodesic metric in latent
  space for efficient and stable sampling in medical image-to-image synthesis.
* [Cross-conditioned Diffusion Model for Medical Image to Image Translation (2024)](https://arxiv.org/abs/2409.08500) (paper only).
  Multi-modal MRI (T1, T2, FLAIR; public datasets). Image-to-Image method. Cross-modality conditioning where the source MRI guides
  target-modality diffusion; modality-specific encoders enhance structural and contrast fidelity.
* [GitHub: Cascaded diffusion models for medical image translation](https://github.com/ycaris/Cascaded-Multi-Path-Diffusion-Medical-Image-Translation)
  [paper link](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002251). Brain / Cardiac (general datasets).
  Image-to-Image method. Combines a coarse GAN prior with a diffusion refinement stage; shortcut paths reduce steps while preserving
  fidelity and uncertainty quantification.
* [Fast-DDPM: Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation (JBHI 2025)](https://github.com/mirthAI/Fast-DDPM) (GitHub and paper link).
  General (denoising, SR, modality transfer). MRI / CT (Brain, Thorax; open datasets). Image-to-Image method. Efficient DDPM variant using only 10 diffusion steps;
  achieves state-of-the-art results on denoising, super-resolution, and modality translation tasks.


# Generative AI Video Models
* [EchoNet-Synthetic: Privacy-preserving Video Generation for Safe Medical Data Sharing](https://github.com/HReynaud/EchoNet-Synthetic) and [paper](https://arxiv.org/abs/2406.00808)  (also see Echonet datasets and models).
* [Endora: Video Generation Models as Endoscopy Simulators](https://github.com/CUHK-AIM-Group/Endora)
* [A multimodal video dataset of human spermatozoa](https://www.kaggle.com/datasets/stevenhicks/visem-video-dataset)
* [A public endoscopic video dataset for polyp detection](https://github.com/dashishi/LDPolypVideo-Benchmark)
* [Carotid Ultrasound Boundary Study (CUBS): Technical considerations on an open multi-center analysis of computerized measurement systems for intima-media thickness measurement on common carotid artery longitudinal B-mode ultrasound scans](https://data.mendeley.com/datasets/m7ndn58sv6/1)
  
# Tensorflow models for video and multimodal risk assessment 
* [DISIML models: Echo, ECG, tabular data models, and autoencoders for dimensionality reduction](https://alvarouc.gitlab.io/disiml/)
* [A Large-scale Multimodal Study for Predicting Mortality Risk Using Minimal and Low Parameter Models and Separable Risk Assessment](https://ieeexplore.ieee.org/abstract/document/10839321)
* [Deep-learning-assisted analysis of echocardiographic videos improves predictions of all-cause mortality](https://www.nature.com/articles/s41551-020-00667-9)

# Open Models for Explainability
* [Gradcam: Advanced AI explainability for PyTorch](https://github.com/jacobgil/pytorch-grad-cam)
* [HiResCAM: A small demo of the HiResCAM and Grad-CAM gradient-based neural network explanation methods](https://github.com/rachellea/hirescam)
* [Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks](https://github.com/adityac94/Grad_CAM_plus_plus)
* [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](https://github.com/haofanwang/Score-CAM)
* [Ablation-CAM: Visual Explanations for Deep Convolutional Network via Gradient-free Localization](https://github.com/ShreyPandit/Ablation-Cam)
* [Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs](https://github.com/Fu0511/XGrad-CAM)
* [LayerCAM: Exploring Hierarchical Class Activation Maps for Localization](https://github.com/mhyatt000/layerCAM)

# PyTorch Video Models, Datasets, and Optimization Resources
## [PyTorch video resources](https://pytorchvideo.org/)
* [Models documentation](https://pytorchvideo.readthedocs.io/en/latest/models.html)
* [Models on GitHub](https://github.com/facebookresearch/pytorchvideo/tree/main/pytorchvideo/models/hub)
* [Pretrained models on specific datasets and performance](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md)
* [Build your own model tutorial](https://pytorchvideo.org/docs/tutorial_accelerator_build_your_model#introduction)

## Select PyTorch image and video classification models
* [Vision ResNet model](https://docs.pytorch.org/vision/stable/models/video_resnet.html)
* [3D ResNet](https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/)
* [X3D: Expanding Architectures for Efficient Video Recognition](https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/)
* [SlowFast Networks for Video Recognition](https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/)
* [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://docs.pytorch.org/vision/main/models/video_mvit.html)
* [Video Swin Transformer model](https://docs.pytorch.org/vision/stable/models/video_swin_transformer.html)

## PyTorch video documentation
* [Video datasets](https://docs.pytorch.org/vision/main/datasets.html#video-classification)
* [Optical Flow datasets for video motion estimation](https://docs.pytorch.org/vision/main/datasets.html#optical-flow)

## [Main optimization link in PyTorch](https://docs.pytorch.org/docs/stable/optim.html)
* [PyTorch optimization methods](https://docs.pytorch.org/docs/stable/optim.html#algorithms).
* [PyTorch vision models training parameters](https://github.com/pytorch/vision/tree/main/references/classification)

## [Pytorch: Adjusting the learning rate](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
* [A (Very Short) Visual Introduction to Learning Rate Schedulers (With Code)](https://medium.com/@theom/a-very-short-visual-introduction-to-learning-rate-schedulers-with-code-189eddffdb00)
* [Step learning rate scheduler](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
* [Reduce the learning rate when we reach a plateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) 

## Model Evaluation Notes
For evaluating your models, consider
   [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning by Sebastian Raschka.](https://arxiv.org/abs/1811.12808).

## How to find other datasets and models from general purpose websites
1. Search for Datasets on [Google Dataset Search](https://datasetsearch.research.google.com/).
2. Search for [Papers with code](https://paperswithcode.com/). Look separately for Methods and Datasets.
3. Search for datasets, models, and dataset competitions on [kaggle](https://www.kaggle.com/).
4. Search for Computer Vision datasets on [PyTorch vision datasets website](https://pytorch.org/vision/stable/datasets.html).
5. Search for pretrained PyTorch models [PyTorch models website](https://pytorch.org/vision/stable/models.html).   
