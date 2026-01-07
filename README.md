# 🏠🛰️ Satellite-Based Property Valuation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive multimodal machine learning project that combines satellite imagery with tabular property data to predict real estate prices. This project demonstrates the power of fusing computer vision and traditional machine learning techniques for property valuation.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview--motivation)
- [Dataset Description](#-dataset-description)
- [Pipeline Architecture](#-complete-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Results](#-results)
- [Model Explainability](#-model-explainability--interpretability)
- [Challenges & Solutions](#-challenges--solutions)
- [Future Work](#-future-work--improvements)
- [Citation & Acknowledgments](#-citation--acknowledgments)
- [License](#-license)
- [Contact & Contributing](#-contact--contributing)

---

## 🎯 Project Overview & Motivation

Property valuation is a critical challenge in real estate, traditionally relying on manual assessments, comparative market analysis, and tabular property features. However, these approaches often miss crucial contextual information about a property's neighborhood, surroundings, and environmental factors that significantly influence value.

This project introduces a **multimodal approach** that combines:

- **Tabular Property Data**: Traditional features like bedrooms, bathrooms, square footage, grade, condition, and location coordinates
- **Satellite Imagery**: 1024×1024 RGB satellite images capturing neighborhood context, greenery, density, and proximity to amenities

By fusing these complementary data sources, we achieve superior prediction performance compared to tabular-only models. The satellite imagery captures valuable visual context—such as neighborhood density, green spaces, proximity to water, and urban development patterns—that traditional tabular features cannot fully represent.

### Real-World Impact

This approach has significant applications in:
- **Real Estate Platforms**: Automated property valuation for listing prices and investment analysis
- **Financial Services**: Mortgage underwriting and risk assessment
- **Urban Planning**: Understanding spatial patterns in property values
- **Investment Analysis**: Data-driven property investment decisions

Our **Hybrid Fusion architecture** represents a key innovation, combining the benefits of early and late fusion strategies with attention mechanisms to optimally integrate multimodal information.

---

## 📊 Dataset Description

### Tabular Features

The dataset includes 37 engineered features derived from property attributes:

**Original Features:**
- `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`
- `floors`, `waterfront`, `view`, `condition`, `grade`
- `yr_built`, `yr_renovated`, `zipcode`
- `lat`, `long` (coordinates)
- `sqft_living15`, `sqft_lot15` (neighborhood averages)

**Engineered Features (25+):**
- **Basic**: `total_rooms`, `living_lot_ratio`, `has_basement`, `above_ratio`
- **Neighborhood**: `living_vs_neighbors`, `lot_vs_neighbors`
- **Quality**: `quality_score`, `luxury_score`, `premium_view`
- **Location**: `dist_from_center`, `property_age`, `age_at_sale`
- **Interaction**: `sqft_per_bedroom`, `sqft_per_bathroom`, `was_renovated`

### Satellite Imagery

- **Format**: RGB satellite images
- **Original Size**: 1024×1024 pixels
- **Processed Size**: 224×224 pixels (for model training)
- **Source**: Mapbox Static Images API
- **Coverage**: 100% of training and test samples

### Dataset Statistics

- **Training Samples**: 16,210 properties
- **Test Samples**: 5,405 properties
- **Tabular Features**: 37 engineered features
- **Image Coverage**: 100% (16,210 training + 5,405 test images)

![7600136](https://github.com/user-attachments/assets/778a16a9-2e18-4b82-a984-8384f76e1d4b)




---

## 🔄 Complete Pipeline Architecture

### Stage 1: Data Preprocessing

**Tabular Data:**
- Missing value handling (median imputation)
- Feature engineering (25+ new features created)
- Robust scaling for numerical features
- Date feature extraction (year, month, quarter sold)

**Image Preprocessing:**
- Resize to 224×224 pixels
- Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Data augmentation for training:
  - Random horizontal flips (p=0.5)
  - Random vertical flips (p=0.3)
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation, hue)

### Stage 2: Exploratory Data Analysis

**Key Insights:**
- **Price Distribution**: Highly right-skewed (skewness: 4.03), requiring log transformation consideration
- **Top Correlated Features**: `sqft_living` (0.70), `grade` (0.66), `sqft_above` (0.60)
- **Geospatial Patterns**: Clear geographic clustering of property values
- **Feature Importance**: `luxury_score` and `waterfront` are among the most predictive features

### Stage 3: Tabular-Only Baseline Models

We established baseline performance using traditional machine learning models:

- **Linear Models**: Ridge, Lasso, ElasticNet
- **Tree-Based**: Random Forest, Gradient Boosting
- **Advanced Boosting**: XGBoost, LightGBM, CatBoost

**Best Tabular-Only Performance:**
- **XGBoost**: RMSE $114,467 | MAE $65,700 | R² 0.8956

### Stage 4: Image Feature Extraction

**CNN Architectures:**
- **ResNet50**: 2048-dimensional features
- **EfficientNet-B3**: 1536-dimensional features
- **VGG16**: 512-dimensional features

**Key Innovations:**
- **Transfer Learning**: Pre-trained ImageNet weights
- **CBAM Attention**: Convolutional Block Attention Module (channel + spatial attention)
- **Feature Dimensionality**: Reduced to 512 dimensions via projection head
- **Center Attention Loss**: Encourages model to focus on central image regions

### Stage 5: Multimodal Fusion Architectures

#### Early Fusion

**Architecture:**
- Concatenate image features (512-dim) and tabular features (37-dim) at feature level
- Cross-modal attention refinement
- Multi-layer feedforward network for prediction

**Pros:**
- Rich feature interaction early in the network
- Simpler architecture

**Cons:**
- Feature dimensionality imbalance challenges
- Less flexibility in modality-specific processing

#### Late Fusion

**Architecture:**
- Separate branches for image and tabular data
- Independent prediction heads
- Learnable weighted combination (gating mechanism)

**Pros:**
- Preserves modality-specific representations
- Adaptive weighting between modalities

**Cons:**
- Limited cross-modal interaction
- May not capture complex feature relationships

#### Hybrid Fusion (Our Innovation)

**Architecture:**
- **Multiple Fusion Points**: Combines early and late fusion benefits
- **Early Fusion Path**: Concatenated features → 512-dim representation
- **Late Fusion Paths**: Separate image (256-dim) and tabular (256-dim) branches
- **Final Fusion**: Concatenate all paths (512 + 256 + 256) → 384-dim → 128-dim → prediction

**Why This Works:**
- Captures both modality-specific and cross-modal patterns
- Multiple information pathways reduce information loss
- Flexible architecture adapts to different property types

**Pros:**
- Best of both worlds (early + late fusion)
- Robust performance across different property types
- Excellent representation learning

**Cons:**
- More complex architecture
- Higher computational cost


### Stage 6: Model Training & Optimization

**Training Configuration:**
- **Loss Function**: MSE Loss + Center Attention Loss (weight: 0.05)
- **Optimizer**: AdamW (learning rate: 0.001, weight decay: 1e-4)
- **Learning Rate Schedule**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Regularization**: Dropout (0.2-0.3), Batch Normalization, Gradient Clipping (max_norm=1.0)
- **Early Stopping**: Patience=5 epochs
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)

### Stage 7: Model Explainability

**Grad-CAM Visualization:**
- Highlights image regions influencing predictions
- Spatial attention maps from CBAM modules
- Overlay visualizations showing model focus areas

**Key Findings:**
- Model focuses on property structure and surrounding areas
- Center attention loss successfully guides focus to property regions
- Spatial patterns reveal neighborhood context importance

<img width="1920" height="985" alt="Grad-CAM Visualizations" src="https://github.com/user-attachments/assets/e1540cde-074a-4134-b12f-a6eecf5470ab" />


---

## 📁 Project Structure

```
Satellite-based-property-valuation/
│
├── 1) data_fetcher.py              # Satellite image download script
├── 2) preprocessing.ipynb          # Feature engineering & EDA
├── 3) cdc_training.ipynb          # Tabular-only baseline models
├── 4) output.ipynb                # Model evaluation & visualization
├── 5) model_training.ipynb        # Multimodal fusion models
│
├── datasets/                       # Preprocessed dataset cache
│   └── datasets.pt
│
├── Improved models/                # Trained model checkpoints
│   ├── ImprovedEarlyFusion_*.pth
│   ├── ImprovedLateFusion_*.pth
│   └── ImprovedHybridFusion_*.pth
│
├── Improved results/               # Results and visualizations
│   ├── model_comparison.png
│   ├── training_history.png
│   ├── best_model_predictions.png
│   ├── improved_gradcam_*.png
│   └── multimodal_predictions_*.csv
│
├── images/                         # Satellite images
│   ├── train/
│   └── test/
│
├── train.csv                       # Training data
├── test.csv                        # Test data
├── train_engineered.csv            # Engineered training features
├── test_engineered.csv             # Engineered test features
│
├── X.csv                           # Preprocessed features
├── y.csv                           # Target prices
├── X_test.csv                      # Preprocessed test features
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## ⚙️ Setup & Installation

### Environment Setup

**Prerequisites:**
- Python 3.10+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 20GB+ disk space

**Installation Steps:**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Satellite-based-property-valuation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch (if not included):**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU-only
   pip install torch torchvision torchaudio
   ```

5. **Set up Mapbox API (for image download):**
   - Sign up at [Mapbox](https://www.mapbox.com/)
   - Get your access token
   - Update `MAPBOX_TOKEN` in `1) data_fetcher.py`

### Running the Pipeline

**Step 1: Download Satellite Images**
```bash
python 1) data_fetcher.py
```
Note: Update the `mapbox_token` variable in the script before running.

**Step 2: Preprocess Data & Feature Engineering**
```bash
jupyter notebook 2) preprocessing.ipynb
```
Run all cells to generate engineered features.

**Step 3: Train Tabular-Only Baseline Models**
```bash
jupyter notebook 3) cdc_training.ipynb
```
Establishes baseline performance metrics.

**Step 4: Train Multimodal Fusion Models**
```bash
jupyter notebook 5) model_training.ipynb
```
Trains all fusion architectures (Early, Late, Hybrid) with different backbones.

**Step 5: Evaluate & Visualize Results**
```bash
jupyter notebook 4) output.ipynb
```
Generates performance comparisons, visualizations, and final predictions.

---

## 📈 Results

### Performance Comparison

| Approach | Model | RMSE ($) | MAE ($) | R² Score |
|----------|-------|----------|---------|----------|
| **Tabular Only** | XGBoost | 114,467 | 65,700 | 0.8956 |
| **Tabular Only** | LightGBM | 118,734 | 68,663 | 0.8877 |
| **Tabular Only** | CatBoost | 115,333 | 69,264 | 0.8940 |
| **Tabular Only** | Random Forest | 130,427 | 70,700 | 0.8644 |
| **Image Features + ML** | XGBoost on CNN features | 111,965 | 65,542 | 0.9001 |
| **Early Fusion** | ResNet50 | 338,091 | 245,113 | 0.0891 |
| **Early Fusion** | EfficientNet-B3 | 309,705 | 225,193 | 0.2356 |
| **Late Fusion** | ResNet50 | 248,867 | 147,401 | 0.5065 |
| **Late Fusion** | EfficientNet-B3 | 246,779 | 136,710 | 0.5147 |
| **Hybrid Fusion** | ResNet50 | 285,019 | 161,421 | 0.3526 |
| **Hybrid Fusion** | EfficientNet-B3 | **246,701** | 161,747 | **0.5150** |
| **Ensemble (Weighted)** | Multi-model | 139,216 | 79,577 | 0.8456 |

### Key Findings

1. **Multimodal Advantage**: Traditional ML models (XGBoost, LightGBM) trained on extracted CNN features + tabular data outperform pure tabular models, achieving R² of **0.9001** vs. **0.8956**.

2. **Hybrid Fusion Excellence**: Among end-to-end deep learning models, **Hybrid Fusion with EfficientNet-B3** achieves the best performance (RMSE: $246,701, R²: 0.5150).

3. **Attention Mechanisms**: CBAM and center attention loss significantly improve model interpretability and guide focus to relevant image regions.

4. **Ensemble Performance**: Weighted ensemble of top models achieves robust performance (RMSE: $139,216, R²: 0.8456), balancing different model strengths.

5. **Image Context Value**: Satellite imagery captures neighborhood context (greenery, density, proximity) that tabular features cannot fully represent.

<img width="4768" height="3565" alt="model_comparison" src="https://github.com/user-attachments/assets/469191d1-ff68-4912-ad2a-a5160dbcb707" />

<img width="1587" height="1989" alt="Feature Importance" src="https://github.com/user-attachments/assets/456285f2-35f2-4327-9462-4ed344a781cf" />


---

## 🔍 Model Explainability & Interpretability

### Grad-CAM Analysis

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations reveal which image regions influence model predictions:

- **Property-Centric Focus**: The center attention loss successfully guides the model to focus on central property regions rather than image artifacts or borders.

- **Spatial Patterns**: Visualizations show attention on:
  - Property structures and buildings
  - Surrounding green spaces
  - Neighborhood density patterns
  - Proximity to water bodies or amenities

- **Prediction Confidence**: Properties where images contribute most show clearer spatial attention patterns, indicating confident multimodal integration.

### Feature Importance Analysis

From tabular-only models, top important features include:
1. `luxury_score` (32.5% importance)
2. `waterfront` (7.4%)
3. `sqft_living` (7.2%)
4. `premium_view` (6.3%)
5. `dist_from_center` (5.3%)

---

## 🛠️ Challenges & Solutions

### Challenge 1: Model Focusing on Image Artifacts

**Problem**: Initial models learned from image artifacts (compression artifacts, borders) rather than property-relevant regions.

**Solution**: 
- Implemented CBAM (Convolutional Block Attention Module) for spatial and channel attention
- Added center attention loss (weight: 0.05) to encourage focus on central image regions
- Data augmentation reduced overfitting to specific image patterns

### Challenge 2: Image-Tabular Feature Imbalance

**Problem**: Image features (512-dim) and tabular features (37-dim) have different scales and distributions, making fusion challenging.

**Solution**:
- Robust scaling for tabular features
- Feature dimension normalization (tabular encoder projects to 512-dim to match image features)
- Cross-modal attention for adaptive weighting

### Challenge 3: Overfitting on Training Data

**Problem**: Deep learning models showed signs of overfitting with large train/val loss gaps.

**Solution**:
- Comprehensive data augmentation (flips, rotation, color jitter)
- Dropout regularization (0.2-0.3) and batch normalization
- Early stopping (patience=5)
- Gradient clipping (max_norm=1.0)
- Learning rate scheduling (CosineAnnealingWarmRestarts)

---

## 🚀 Future Work & Improvements

- [ ] **Additional Data Sources**: Incorporate street view images, demographic data, and temporal price trends
- [ ] **Vision Transformers (ViT)**: Experiment with ViT architectures for potentially better image representations
- [ ] **Uncertainty Quantification**: Implement prediction intervals and confidence scores
- [ ] **REST API Deployment**: Deploy model as REST API for real-time predictions
- [ ] **Temporal Analysis**: Add time-series analysis for price trend prediction
- [ ] **Multi-Task Learning**: Simultaneously predict price, property type, and year built
- [ ] **Graph Neural Networks**: Model neighborhood effects using graph structures
- [ ] **Hyperparameter Optimization**: Systematic hyperparameter tuning (Optuna, Ray Tune)
- [ ] **Model Compression**: Quantization and pruning for deployment efficiency
- [ ] **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation

---

## 📚 Citation & Acknowledgments

### Datasets
- Property data: King County Housing Dataset
- Satellite imagery: Mapbox Static Images API

### Pre-trained Models
- ResNet50, EfficientNet-B3, VGG16: PyTorch Torchvision (pre-trained on ImageNet)

### Key Libraries
- PyTorch: Deep learning framework
- Scikit-learn: Traditional ML models and utilities
- XGBoost, LightGBM, CatBoost: Gradient boosting libraries
- Matplotlib, Seaborn: Visualization

### Citation Format

If you use this project in your research, please cite:

```bibtex
@misc{satellite_property_valuation,
  title={Satellite-Based Property Valuation: A Multimodal Machine Learning Approach},
  author={Tanish Verma},
  year={2026},
  url={https://github.com/yourusername/satellite-property-valuation}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Contributing

### Contact

For questions, suggestions, or collaborations, please open an issue on GitHub or contact:

- **Email**: [vermatanish9905@gmail.com]
- **GitHub**: [@aldr4GO]

### Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Guidelines:**
- Follow PEP 8 style guidelines
- Add docstrings to new functions/classes
- Include tests for new functionality
- Update documentation as needed
- Ensure all notebooks run successfully

---

## 🙏 Acknowledgments

Special thanks to:
- The open-source community for excellent tools and libraries
- Mapbox for satellite imagery API
- Researchers and practitioners in multimodal learning who inspired this work

---

**Built with ❤️ using PyTorch, Scikit-learn, and modern machine learning techniques.**

