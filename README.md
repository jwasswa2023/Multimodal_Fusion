# Multimodal Fusion for Molecular Property Prediction

A Python-based research repository for predicting molecular properties using multimodal fusion approaches. The project combines molecular representations, MS/MS fragmentation information, and machine learning models to evaluate early fusion, late fusion, uncertainty, and modality contribution.

---

## Project Structure

```text
Multimodal_Fusion/
├── src/
│   ├── fusion_early.py
│   ├── fusion_late.py
│   ├── uncertainty_analysis.py
│   ├── modality_contribution.py
│   ├── Data_processing.py
│   ├── MS2_frag_Processing.py
│   └── utils.py
├── notebooks/
│   ├── 01_Data_processing.ipynb
│   ├── 02_MS2_frag_Processing.ipynb
│   ├── 03_Early_Fusion.ipynb
│   ├── 04_Late_Fusion.ipynb
│   ├── 05_Uncertainty.ipynb
│   └── 06_Modality_Contribution.ipynb
├── data/
├── requirements.txt
├── LICENSE
└── README.md




# Features

```markdown
## Features

- Data preprocessing for molecular property prediction  
- MS/MS fragmentation processing  
- Early fusion modeling  
- Late fusion modeling  
- Uncertainty analysis  
- Modality contribution analysis  
- Reproducible notebook workflows  

## Installation

Clone the repository:

```bash
git clone https://github.com/jwasswa2023/Multimodal_Fusion.git
cd Multimodal_Fusion

### Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate

### On Windows:
venv\Scripts\activate


### Install dependencies:

pip install -r requirements.txt


```markdown
## Requirements

This project uses:

- Python  
- NumPy  
- pandas  
- scikit-learn  
- LightGBM  
- PyTorch  
- PyTorch Geometric  
- DeepChem  
- RDKit  
- SHAP  
- DGL  
- DGL-LifeSci  
- mol2vec  
- matplotlib  


## Workflow

Run the notebooks in order:

1. 01_Data_processing.ipynb  
2. 02_MS2_frag_Processing.ipynb  
3. 03_Early_Fusion.ipynb  
4. 04_Late_Fusion.ipynb  
5. 05_Uncertainty.ipynb  
6. 06_Modality_Contribution.ipynb


## Methods

This repository explores multimodal molecular property prediction using:

### Early Fusion
Combines multiple molecular feature representations before model training.

### Late Fusion
Trains separate models for different modalities and combines predictions afterward.

### Uncertainty Analysis
Evaluates confidence and reliability of model predictions.

### Modality Contribution
Assesses the contribution of each molecular data modality to prediction performance.


## License

This project is licensed under the MIT License.

## Acknowledgments

Artificial intelligence tools from OpenAI, Anthropic, and Google were used as coding aids during development.

## Contact

Joseph Wasswa  
SUNY Polytechnic Institute  
wasswaj@sunypoly.edu  
