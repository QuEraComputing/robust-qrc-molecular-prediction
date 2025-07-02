# Quantum Reservoir Computing (QRC) for Molecular Activity Prediction

This repository contains the code and data processing scripts for reproducing the results presented in our paper:

**"Robust Quantum Reservoir Computing for Molecular Property Prediction"**  
*Daniel Beaulieu, Milan Kornjaca, Zoran Krunic, Michael Stivaktakis, Thomas Ehmer, Sheng-Tao Wang, Anh Pham*  
arXiv:2412.06758

Paper link: [https://arxiv.org/abs/2412.06758](https://arxiv.org/abs/2412.06758)

## Overview

This project implements Quantum Reservoir Computing (QRC) and Classical Reservoir Computing (CRC) methods for molecular activity prediction using the Merck Molecular Activity Challenge dataset. The codebase includes data preprocessing, embedding generation, model training, and evaluation scripts.

## Prerequisites

### Python Environment

- Python dependencies are managed using `uv`
- Install dependencies: `uv sync`
- Configuration files: `pyproject.toml` and `uv.lock`

### Julia Environment

- Julia packages are specified in `Manifest.toml`
- All necessary Julia packages are included in the project manifest

## Dataset

Download the Merck Molecular Activity Challenge dataset (2.1GB) from [Kaggle](https://www.kaggle.com/competitions/MerckActivity/overview) and place the files in `DATA/TrainingSet/`. The dataset files are named as `ACT{number}_competition_training.csv`.

## Usage

### 1. Data Preparation

Run `qrc-dataprep.py` to create subsamples for 100, 200, and 800 records, as well as 25 subsamples of 100 samples for cross-validation.  The script implements the data preparation pipeline and classical regression models used in the paper. It incorporates data cleaning, outlier detection, feature scaling, dimensionality reduction (PCA), and feature selection using SHAP values. It generates multiple stratified subsamples and cross-validation splits, enabling robust evaluation of machine learning models on both original and QRC-embedded features. The workflow supports reproducible experiments by saving processed datasets and feature selections, facilitating systematic comparison of classical and quantum-inspired approaches in large-scale molecular regression tasks.

### 2. Generate Embeddings

#### QRC Embeddings

Run `qrc_regression_merck.jl` to create QRC embeddings and QRC one-body embeddings from the preprocessed data. The script encodes classical molecular descriptors into quantum states using quantum reservoir detuning layers and simulates their evolution under a Rydberg Hamiltonian, extracting quantum observables as high-dimensional embeddings. It processes subsamples for a given data set a a given number of records (100,200,800) and different experimental configurations.  The QRC embedding outputs are used in `qrc_runalgos_alltypes.py`.  

#### CRC Embeddings

Run `crc_randforest_embeddingonly.jl` to generate Classical Reservoir Computing (CRC) embeddings. Classical reservoir embeddings are generated from vector spin limit of the Rydberg Hamiltonian. The script processes molecular activity datasets by normalizing and projecting features, simulating their evolution through the integration of the spin dynamical system, and extracts time-dependent observables as embeddings. These embeddings are saved for downstream machine learning tasks to systematically compare CRC-based representations with quantum and classical baselines. The code is designed to process subsamples for a particular Merck molecular activity dataset, a number of data subsamples, a specified number of records (100,200,800), and various reservoir configurations. The CRC embedding outputs are used in `qrc_runalgos_alltypes_crc.py`.  

#### Shot Noise Simulation Embeddings

Run `qrc_regression_wavefunction_milan.jl` to create embeddings using simulated shot noise through the wavefunction sampling. The script encodes classical molecular descriptors into quantum states via quantum reservoir detuning layers, simulates their evolution under a Rydberg Hamiltonian, and extracts measurement observables from a finite number of wavefunction samples as high-dimensional quantum embeddings, equivalently to `qrc_regression_merck.jl`. The user can specify the number of samples, with 0 representing exact sampling. The data collection for the paper is run repeatedly for each subsample in order to estimate shot noise related uncertainty.

### 3. Model Training and Evaluation

#### Standard Models

Run `qrc_runalgos_alltypes.py` to train and evaluate models on classical data, QRC embeddings, and QRC one-body embeddings. This script code runs the modeling pipeline and creates regression model diagnostics statistics such as mean-squared error, accuracy, AUC, F1-Score, recall, and precision for classical data, QRC two-body embeddings, and QRC one-body embeddings. It automates data postprocessing and model training using a range of scikit-learn regressors and neural networks. The script evaluates models on both raw molecular features and quantum reservoir computing (QRC) embeddings, systematically comparing performance metrics across multiple data splits and embedding types. Results are aggregated for analysis, enabling at scale evaluation of QRC and classical approaches.

#### CRC Models

Run `qrc_runalgos_alltypes_crc.py` to evaluate models specifically for CRC embeddings. This Python code runs the model pipeline for CRC embeddings and creates regression model diagnostics statistics equivalent to `qrc_runalgos_alltypes.py`.

#### Noise Simulation Models

Run `qrc_runalgos_alltypes_noise.py` to evaluate models for wavefunction sampling embeddings with various noise levels. This Python code runs the model pipeline for the shot noise simulated QRC embeddings and creates regression model diagnostics statistics equivalent to `qrc_runalgos_alltypes.py`.

### 4. Visualization and Analysis

#### UMAP Analysis

Run `merck_activity_QRC_UMAP_recs200-sub4-act4_wbintargs_v3.ipynb` to generate UMAP visualizations and statistical analysis. The notebook loads the QRC and classical embeddings created in the previous script, including QRC, classical reservoir, and  classical reservoir computing (CRC) embedding. The program uses UMAP to performs topological data analysis and visualises low dimensional UMAP representations.

#### Figure Generation

Run `Merck_boxplot.ipynb` to create figures and tables used in the paper.

## File Structure

### Core Scripts

- `qrc-dataprep.py` - Data preprocessing and subsampling
- `qrc_regression_merck.jl` - QRC embedding generation
- `crc_randforest_embeddingonly.jl` - CRC embedding generation
- `qrc_regression_wavefunction_milan.jl` - Noise simulation embeddings
- `qrc_runalgos_alltypes.py` - Model pipeline for standard embeddings
- `qrc_runalgos_alltypes_crc.py` - Model pipeline for CRC embeddings
- `qrc_runalgos_alltypes_noise.py` - Model pipeline for noise simulations

### Notebooks

- `merck_activity_QRC_UMAP_recs200-sub4-act4_wbintargs_v3.ipynb` - UMAP analysis
- `Merck_boxplot.ipynb` - Paper figure generation

### Generated Data Files

All processed data files are stored in the `DATA/` folder with the following naming convention:

```text
{x|y}_{train|test}_{N}rec_sub{M}act{K}v{V}.csv
```

Where:

- `x` = input features, `y` = labels
- `N` = number of records in subsample (e.g., 200)
- `M` = subsample index (1-25 for cross-validation)
- `K` = activity dataset number (e.g., ACT4)
- `V` = version tag

## Key Parameters

### Important Variables

- **`nfeats`**: Number of features selected for QRC by SHAP. Significantly impacts computational time. Reduce for testing.
- **`n_cluster`**: Number of clusters for sampling. Ensures comprehensive coverage during sampling.
- **`shapsample`**: Number of observations used in SHAP analysis. Affects computational cost and memory usage.
- **`recs`**: Number of records in each subsample (e.g., 100, 200, 800).
- **`actfile`**: Activity dataset identifier (e.g., ACT4, ACT5, etc.).
- **`version`**: Version tag for generated data files.
- **`subnum`**: Subsample index number (1-25 for cross-validation splits).

## Notes

- All Python and Julia files have been tested and verified to run successfully
- Running scripts with identical parameters will generate different results due to random sampling
- If you encounter "file not found" errors for CSV files, run the data generation scripts first
- For parameter modifications or reruns, regenerate data using the source scripts before plotting
- **Performance optimization**: If computations are too slow, reduce `nfeats` (affects QRC computational complexity) or `shapsample` (affects SHAP process computational complexity)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{beaulieu2024robust,
  title={Robust Quantum Reservoir Computing for Molecular Property Prediction},
  author={Beaulieu, Daniel and Kornjaca, Milan and Krunic, Zoran and Stivaktakis, Michael and Ehmer, Thomas and Wang, Sheng-Tao and Pham, Anh},
  journal={arXiv preprint arXiv:2412.06758},
  year={2024},
  url={https://arxiv.org/abs/2412.06758}
}
```


