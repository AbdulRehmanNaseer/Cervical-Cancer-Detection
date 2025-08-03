# FNC Ensemble CNN Models for Cervical Cancer Detection

## Abstract

Cervical cancer remains one of the most prevalent and deadly cancers affecting women worldwide, with early detection being crucial for successful treatment outcomes. This project presents an innovative ensemble approach using multiple Convolutional Neural Network (CNN) architectures combined with advanced attention mechanisms to achieve highly accurate cervical cell classification. The proposed FNC (Fusion of Neural Classifiers) ensemble model leverages the complementary strengths of EfficientNetB0, MobileNet, and ResNet50 architectures, enhanced with channel and spatial attention mechanisms, to achieve an impressive accuracy of **97.16%** on the SIPAKMED dataset. This approach demonstrates superior performance in distinguishing between five critical cervical cell types: Parabasal, Dyskeratotic, Metaplastic, Superficial-Intermediate, and Koilocytotic cells, making it a valuable tool for automated cervical cancer screening and diagnosis.

## Problem Statement

Cervical cancer screening traditionally relies on manual examination of cervical cell images by cytopathologists, which is time-consuming, subjective, and prone to human error. The challenge lies in accurately classifying different types of cervical cells from microscopic images, as these classifications are critical indicators of cervical health and potential malignancy. Traditional methods face several limitations:

- **Subjectivity**: Manual interpretation varies between pathologists
- **Time-intensive**: Manual screening requires significant expertise and time
- **Scalability issues**: Limited availability of expert cytopathologists
- **Consistency**: Inter-observer and intra-observer variability in interpretations

The need for automated, reliable, and scalable solutions for cervical cell classification has become increasingly important for improving screening programs and reducing the burden on healthcare systems.

## Dataset: SIPAKMED

The project utilizes the **SIPAKMED** (Sipakmed Pap Smear) dataset, a comprehensive collection of cervical cell images designed for automated cytology analysis. The dataset contains 4,049 high-quality microscopic images of cervical cells, categorized into five distinct classes:

| Cell Type | Number of Images | Description |
|-----------|------------------|-------------|
| **Parabasal** | 787 | Small, round cells with dense nuclei, typically found in atrophic conditions |
| **Dyskeratotic** | 813 | Abnormal cells with premature keratinization, potential cancer indicators |
| **Metaplastic** | 793 | Squamous metaplasia cells, normal response to irritation |
| **Superficial-Intermediate** | 831 | Mature squamous cells, normal cervical epithelium |
| **Koilocytotic** | 825 | Cells with perinuclear halos, associated with HPV infection |

### Dataset Characteristics:
- **Total Images**: 4,049
- **Image Format**: RGB (224×224 pixels)
- **Classes**: 5 distinct cervical cell types
- **Distribution**: Relatively balanced across classes
- **Quality**: High-resolution microscopic images

## Approach: FNC Ensemble Model

### Model Architecture

The FNC ensemble approach combines three state-of-the-art CNN architectures with advanced attention mechanisms:

#### 1. **Base Models**
- **EfficientNetB0**: Efficient architecture with compound scaling
- **MobileNet**: Lightweight model optimized for mobile deployment
- **ResNet50**: Deep residual network with skip connections

#### 2. **Ensemble Fusion Strategy**
The ensemble employs a sophisticated fusion approach:

```python
# Feature Extraction
efficientnetb0_output = GlobalAveragePooling2D()(efficientnetb0.output)
mobilenet_output = GlobalAveragePooling2D()(mobilenet.output)
resnet_output = GlobalAveragePooling2D()(resnet.output)

# Feature Concatenation
concatenated_features = Concatenate(axis=-1)([
    efficientnetb0_output, 
    mobilenet_output, 
    resnet_output
])
```

#### 3. **Attention Mechanisms**

**Channel Attention**: Focuses on "what" to pay attention to by learning channel-wise dependencies:
- Global Average Pooling and Global Max Pooling
- Shared MLP with reduction ratio
- Sigmoid activation for attention weights

**Spatial Attention**: Focuses on "where" to pay attention by learning spatial relationships:
- Channel-wise average and max pooling
- Convolutional layer with 7×7 kernel
- Sigmoid activation for spatial attention map

**Channel-Spatial Attention**: Sequential application of both attention mechanisms for comprehensive feature refinement.

#### 4. **Advanced Processing Layers**
- **Depthwise Separable Convolutions**: Efficient feature extraction
- **Batch Normalization**: Training stability
- **Dropout (0.5)**: Regularization to prevent overfitting
- **L2 Regularization**: Additional regularization
- **Exponential Learning Rate Decay**: Adaptive learning rate scheduling

### Training Configuration

- **Optimizer**: Adam with exponential decay
- **Learning Rate**: 1e-5 with decay steps of 10,000
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50
- **Train/Test Split**: 80/20 with stratification
- **Validation Split**: 20% of training data

## Results and Performance

### Overall Performance
- **Accuracy**: 97.16%
- **Macro Average F1-Score**: 97.17%
- **Weighted Average F1-Score**: 97.16%

### Per-Class Performance

| Cell Type | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| **Parabasal** | 98.74% | 100.00% | 99.37% | 157 |
| **Dyskeratotic** | 97.53% | 96.93% | 97.23% | 163 |
| **Metaplastic** | 95.60% | 95.60% | 95.60% | 159 |
| **Superficial-Intermediate** | 100.00% | 99.40% | 99.70% | 166 |
| **Koilocytotic** | 93.94% | 93.94% | 93.94% | 165 |

### Key Achievements

1. **High Accuracy**: 97.16% overall accuracy demonstrates excellent classification performance
2. **Balanced Performance**: Consistent performance across all cell types
3. **Robust Classification**: Strong performance on the most critical cell types (Dyskeratotic and Koilocytotic)
4. **Clinical Relevance**: High precision and recall for cancer-indicating cell types

## Technical Implementation

### Dependencies
```python
tensorflow>=2.0.0
numpy
pandas
scikit-learn
keras
```

### Key Features
- **Multi-model Ensemble**: Combines three different CNN architectures
- **Attention Mechanisms**: Channel and spatial attention for feature refinement
- **Advanced Regularization**: Dropout, BatchNorm, and L2 regularization
- **Adaptive Learning**: Exponential learning rate decay
- **Robust Preprocessing**: Image normalization and augmentation

### Model Architecture Highlights
- **Input**: 224×224×3 RGB images
- **Feature Fusion**: Concatenation of three pre-trained model features
- **Attention Processing**: Sequential channel and spatial attention
- **Output**: 5-class softmax classification

## Clinical Significance

This ensemble approach offers several advantages for clinical implementation:

1. **High Reliability**: 97.16% accuracy provides confidence in automated screening
2. **Comprehensive Coverage**: Handles all major cervical cell types
3. **Scalability**: Automated processing reduces manual workload
4. **Consistency**: Eliminates inter-observer variability
5. **Early Detection**: Accurate identification of abnormal cells (Dyskeratotic, Koilocytotic)

## Future Work

Potential areas for further improvement and research:

1. **Real-time Processing**: Optimization for clinical deployment
2. **Additional Cell Types**: Extension to more cell categories
3. **Interpretability**: Attention visualization for clinical validation
4. **Multi-modal Fusion**: Integration with clinical metadata
5. **Transfer Learning**: Adaptation to other cytology datasets

## Conclusion

The FNC ensemble model demonstrates exceptional performance in cervical cell classification, achieving 97.16% accuracy on the SIPAKMED dataset. The combination of multiple CNN architectures with advanced attention mechanisms provides a robust and reliable solution for automated cervical cancer screening. This approach has significant potential for clinical deployment, offering a scalable solution that can improve screening efficiency while maintaining high diagnostic accuracy.

---

**Keywords**: Cervical Cancer Detection, Ensemble Learning, Convolutional Neural Networks, Attention Mechanisms, Medical Image Analysis, SIPAKMED Dataset, Deep Learning, Computer-Aided Diagnosis 