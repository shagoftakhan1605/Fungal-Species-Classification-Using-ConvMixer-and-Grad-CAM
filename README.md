# **Fungal Species Classification Using ConvMixer and Grad-CAM**

## **Overview**
This repository presents a state-of-the-art deep learning pipeline designed to classify fungal species using high-resolution microscopic images. Leveraging the **ConvMixer** architecture and **Gradient-weighted Class Activation Mapping (Grad-CAM)** for interpretability, this work addresses key challenges in fungal classification, including overlapping morphologies, noisy datasets, and the lack of scalable, explainable models. This pipeline is a step toward advancing fungal diagnostics in clinical, agricultural, and ecological contexts.

## **Outputs**
![3](https://github.com/user-attachments/assets/12419343-8042-4c11-be4f-f24e307c10fe)
![4](https://github.com/user-attachments/assets/357e27ca-9722-4951-95fe-1342b0c89841)


---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
    - [ConvMixer Architecture](#convmixer-architecture)
    - [Preprocessing Pipeline](#preprocessing-pipeline)
    - [Model Interpretability](#model-interpretability)
4. [Experimental Results](#experimental-results)
    - [Performance Metrics](#performance-metrics)
    - [Grad-CAM Insights](#grad-cam-insights)
5. [Applications](#applications)
6. [Results](#results)
    - [Training the Model](#training-the-model)
    - [Evaluating the Model](#evaluating-the-model)
    - [Visualizing with Grad-CAM](#visualizing-with-grad-cam)
7. [Future Work](#future-work)
8. [Acknowledgments](#acknowledgments)
9. [License](#license)

---

## **Introduction**
Fungal species classification is a critical task in domains such as clinical diagnostics, agriculture, and ecological monitoring. Traditional diagnostic methods rely on expert mycologists, making the process time-intensive and inaccessible in resource-limited settings. Deep learning models offer an opportunity to automate this task, but they face significant challenges, including:
- **Morphological overlaps** in fungal structures, leading to misclassification.
- **Dataset limitations**, such as noisy images and a lack of diversity in labeled datasets.
- **Interpretability issues** that hinder trust and adoption of AI-driven tools.

This project proposes a novel deep learning pipeline based on the **ConvMixer** architecture, fine-tuned for fungal classification. The pipeline addresses these challenges with:
- A robust preprocessing pipeline tailored for fungal microscopy images.
- The integration of **Grad-CAM** for visualizing model decision-making.
- Comprehensive evaluations of accuracy, precision, recall, F1-score, and computational efficiency.

---

## **Dataset** 
Dataset: (https://www.kaggle.com/datasets/joebeachcapital/defungi/data)

The **DeFungi** dataset, curated for this study, consists of 9,114 high-resolution microscopic image patches labeled into five fungal species:
1. **Tortuous Septate Hyaline Hyphae (TSH):** Morphologically characterized by tortuous, septate, and hyaline hyphae.
2. **Beaded Arthroconidial Septate Hyaline Hyphae (BASH):** Beaded, segmented structures with clear septations.
3. **Groups or Mosaics of Arthroconidia (GMA):** Unique group formations of arthroconidia in mosaic patterns.
4. **Septate Hyaline Hyphae with Chlamydioconidia (SHC):** Septate hyphae with chlamydospores, indicating mold-like fungi.
5. **Broad Brown Hyphae (BBH):** Dark-pigmented, broad hyphae typical of dematiaceous molds.

![H1_1a_17](https://github.com/user-attachments/assets/35de3d71-7a55-4bb8-ad27-e27bd4e79a17)
![H2_1a_8](https://github.com/user-attachments/assets/13e12614-164e-40c3-9c06-b4560dd0668a)
![H3_3b_7](https://github.com/user-attachments/assets/94ebf6d0-0e28-4011-8aa8-6ed0e5042529)
![H5_3a_1](https://github.com/user-attachments/assets/04f2e89c-7704-4263-9193-ca964d0fa653)
![H6_2a_2](https://github.com/user-attachments/assets/740c5f35-c63f-4105-ada0-420df124da8c)


### **Preprocessing**
The dataset underwent extensive preprocessing:
- **Noise Reduction:** Applied median filtering to remove background noise.
- **Artifact Removal:** Used morphological thinning and adaptive thresholding to enhance fungal structures.
- **Standardization:** Resized images to \( 32 \times 32 \) pixels while retaining structural details.

---

## **Methodology**
### **ConvMixer Architecture**
The ConvMixer architecture was selected for its ability to efficiently extract hierarchical features in images. Key components include:
- **Depthwise Convolutions:** For learning spatial patterns in fungal structures.
- **Pointwise Convolutions:** For feature projection and dimensionality reduction.
- **Skip Connections:** To mitigate gradient vanishing and enable efficient training.

The network was fine-tuned on the DeFungi dataset, with the lower layers frozen initially to preserve pre-trained ImageNet features, followed by joint optimization of all layers.

### **Preprocessing Pipeline**
- **Contrast Enhancement:** Used adaptive histogram equalization to highlight fungal morphological features.
- **Patch Extraction:** Divided larger images into \( 500 \times 500 \) pixel patches for consistent input to the model.
- **Normalization:** Scaled pixel values to a \( [0, 1] \) range for faster convergence.

### **Model Interpretability**
Grad-CAM was integrated into the pipeline to generate heatmaps highlighting regions most influential in the model's predictions. These visualizations were validated against expert annotations, demonstrating alignment with biologically relevant fungal features, such as hyphal branching and spore arrangements.

---

## **Experimental Results**
### **Performance Metrics**
The ConvMixer model achieved the following:
- **Accuracy:** \( 85\% \)
- **Precision:** \( 87\% \)
- **Recall:** \( 85\% \)
- **F1-Score:** \( 85\% \)
- **AUC-ROC:** \( 0.95 \)

### **Grad-CAM Insights**
- Heatmaps confirmed the model’s focus on key fungal structures, including septations, spore arrangements, and pigmentation patterns.
- The visual explanations significantly improved trust and interpretability, aligning well with expert assessments.

---

## **Applications**
1. **Clinical Diagnostics:** Early detection and classification of fungal infections to inform antifungal therapy.
2. **Ecological Monitoring:** Tracking fungal biodiversity and their ecological roles in decomposition and nutrient cycling.
3. **Agricultural Management:** Identifying pathogenic fungi affecting crops to enable targeted interventions.

---


## **Results**
### **Training the Model**
![Screenshot (805)](https://github.com/user-attachments/assets/058c27fd-780c-47fc-8506-355e08863f5c)


### **Evaluating the Model**
Evaluate the trained model on the test dataset:
![5](https://github.com/user-attachments/assets/4c07b3fa-d80b-4269-b873-263c93896397)
![7](https://github.com/user-attachments/assets/891758d4-ef30-4d4b-9ece-12eb335c985e)
![Screenshot (807)](https://github.com/user-attachments/assets/b65fdd8c-d6cd-48c0-9725-0fddb2417e0e)


### **Visualizing with Grad-CAM**
![1](https://github.com/user-attachments/assets/df2d29a9-7c6e-4dd5-8fef-593167e39bd4)
![2](https://github.com/user-attachments/assets/84ae441b-6b0b-40eb-8bd9-3ed4a216e25c)



---

## **Future Work**
- **Dataset Expansion:** Including rare and clinically significant fungal species to improve model generalizability.
- **Advanced Interpretability:** Integrating quantitative techniques, such as Layer-wise Relevance Propagation (LRP) or SHAP values.
- **Multimodal Approaches:** Incorporating genomic or biochemical data to enhance classification performance.
- **Edge Deployment:** Optimizing the pipeline for deployment on resource-constrained devices.

---

## **Acknowledgments**
This research was supported by contributions from domain experts in mycology and computational biology. The dataset was generously provided by the DeFungi initiative, with preprocessing techniques informed by best practices in fungal microscopy.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

