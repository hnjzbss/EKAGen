# EKAGen
Code for CVPR2024 paper: "**[Instance-level Expert Knowledge and Aggregate Discriminative Attention for Radiology Report Generation](https://openaccess.thecvf.com/content/CVPR2024/papers/Bu_Instance-level_Expert_Knowledge_and_Aggregate_Discriminative_Attention_for_Radiology_Report_CVPR_2024_paper.pdf)**", Shenshen Bu, Taiji Li, Yuedong Yang, Zhiming Dai. [**[Video](https://www.youtube.com/watch?v=QbcNQ2zuS-8)**]



# Abstract
Automatic radiology report generation can provide substantial advantages to clinical physicians by effectively reducing their workload and improving efficiency. Despite the promising potential of current methods, challenges persist in effectively extracting and preventing degradation of prominent features, as well as enhancing attention on pivotal regions. In this paper, we propose an Instance-level Expert Knowledge and Aggregate Discriminative Attention framework for radiology report generation. We convert expert reports into an embedding space and generate comprehensive representations for each disease, which serve as Preliminary Knowledge Support (PKS). To prevent feature disruption, we select the representations in the embedding space with the smallest distances to PKS as Rectified Knowledge Support (RKS). Then, EKAGen diagnoses the diseases and retrieves knowledge from RKS, creating Instance-level Expert Knowledge (IEK) for each query image, boosting generation. Additionally, we introduce Aggregate Discriminative Attention Map (ADM), which uses weak supervision to create maps of discriminative regions that highlight pivotal regions. For training, we propose a Global Information Self-Distillation (GID) strategy, using an iteratively optimized model to distill global knowledge into EKAGen. Extensive experiments and analyses on IU X-Ray and MIMIC-CXR datasets demonstrate that EKAGen outperforms previous state-of-the-art methods.

<p align="center">
    <img src="docs/EKAGen-framework.png" alt="EKAGen 框架示意图" width="1000" />
</p>

# Get Started

## 1) Requirement

- Python 3.8.13
- Pytorch 1.9.0
- torchvision 0.10.0
- CUDA 11.8
- NVIDIA RTX 4090

## 2) Data Preperation
### MIMIC-CXR
- You must be a credential user defined in [PhysioNet](https://physionet.org/settings/credentialing/) to access the data.
- Download chest X-rays from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and reports from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) Database. 

### IU X-Ray
- You can download the processed reports and images for IU X-Ray by [Chen *et al.*](https://aclanthology.org/2021.acl-long.459.pdf) from [R2GenCMN](https://github.com/cuhksz-nlp/R2GenCMN).

## 3) Download Model Weights and Knowledge Base
* Download the following model weights.
    | Model | Publicly Available |
    | ----- | ------------------- |
    | DiagnosisBot | [diagnosisbot.pth](https://huggingface.co/ShenshenBu/EKAGen/blob/main/diagnosisbot.pth) |
    | ADM Weight | [MIMIC_best_weight.pth](https://huggingface.co/ShenshenBu/EKAGen/blob/main/MIMIC_best_weight.pth) |
    | IU X-Ray Teacher Model | [iu_t_model.pth](https://huggingface.co/ShenshenBu/EKAGen/blob/main/iu_t_model.pth) |
    | MIMIC-CXR Teacher Model | [mimic_t_model.pth](https://huggingface.co/ShenshenBu/EKAGen/blob/main/mimic_t_model.pth) |

* Download the following knowledge base and attention maps.
    | Item | Publicly Available |
    | ----- | ------------------- |
    | IU X-Ray Knowledge Base | [knowledge_prompt_iu.pkl](https://huggingface.co/ShenshenBu/EKAGen/blob/main/knowledge_prompt_iu.pkl) |
    | MIMIC-CXR Knowledge Base | [knowledge_prompt_mimic.pkl](https://huggingface.co/ShenshenBu/EKAGen/blob/main/knowledge_prompt_mimic.pkl) |
    | IU X-Ray ADM | [iu_mask.tar.gz](https://huggingface.co/ShenshenBu/EKAGen/blob/main/iu_mask.tar.gz) |
    | MIMIC-CXR ADM | [mimic_mask.tar.gz](https://huggingface.co/ShenshenBu/EKAGen/blob/main/mimic_mask.tar.gz) |

## 4) Train Models

IU X-Ray
``` bash
bash train_iu.sh
```

MIMIC-CXR
``` bash
bash train_mimic.sh
```

You can also directly download our trained models for inference from [IU X-Ray](https://huggingface.co/ShenshenBu/EKAGen/blob/main/iu_weight.pth) and [MIMIC-CXR](https://huggingface.co/ShenshenBu/EKAGen/blob/main/mimic_weight.pth).

## 5) Test Models

IU X-Ray
``` bash
bash test_iu.sh
```

MIMIC-CXR
``` bash
bash test_mimic.sh
```

## Citation

If you find this work useful in your research, please cite:
```tex
@InProceedings{Bu_2024_CVPR,
    author    = {Bu, Shenshen and Li, Taiji and Yang, Yuedong and Dai, Zhiming},
    title     = {Instance-level Expert Knowledge and Aggregate Discriminative Attention for Radiology Report Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {14194-14204}
}
```
