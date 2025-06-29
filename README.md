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

Run `qrc-dataprep.py` to create subsamples for 100, 200, and 800 records, as well as 25 subsamples of 100 samples for cross-validation.

### 2. Generate Embeddings

#### QRC Embeddings

Run `qrc_regression_merck.jl` to create QRC embeddings and QRC one-body embeddings from the preprocessed data.

#### CRC Embeddings

Run `crc_randforest_embeddingonly.jl` to generate Classical Reservoir Computing embeddings.

#### Noise Simulation Embeddings

Run `qrc_regression_wavefunction_milan.jl` to create embeddings using simulated noise with wavefunction sampling technique.

### 3. Model Training and Evaluation

#### Standard Models

Run `qrc_runalgos_alltypes.py` to train and evaluate models on classical data, QRC embeddings, and QRC one-body embeddings.

#### CRC Models

Run `qrc_runalgos_alltypes_crc.py` to evaluate models specifically for CRC embeddings.

#### Noise Simulation Models

Run `qrc_runalgos_alltypes_noise.py` to evaluate models for wavefunction sampling embeddings with various noise levels.

### 4. Visualization and Analysis

#### UMAP Analysis

Run `merck_activity_QRC_UMAP_recs200-sub4-act4_wbintargs_v3.ipynb` to generate UMAP visualizations and statistical analysis.

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


