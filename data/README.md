# 📊 Sample Data Subset – GSE176078

This repository includes a sample subset of single-cell RNA-seq data derived from the publicly available dataset [GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078). The subset is provided for **demo purposes** and to comply with **GitHub file size limitations**.

## 🧬 Dataset Overview

- **Accession:** [GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078)  
- **Description:** Single-cell transcriptomics of developing human gut and liver (Carnegie stage 11–16)  
- **Source:** NCBI GEO

## 📁 File Provided

- `sample_data_subset.h5ad`: A randomly sampled subset (6,000 cells) of the full dataset in AnnData (`.h5ad`) format.

## 🛠️ How the Subset Was Created

The following Python script was used to generate the sample subset from the full dataset files:

```python
import scanpy as sc
import pandas as pd
import scipy.io
from scipy.sparse import csr_matrix
import numpy as np
import plotly

# Load gene names and barcodes
genes = pd.read_csv('count_matrix_genes.tsv', sep='\t', header=None)[0].values
barcodes = pd.read_csv('count_matrix_barcodes.tsv', sep='\t', header=None)[0].values

# Load sparse matrix and transpose to shape: cells x genes
adata = sc.read_mtx('count_matrix_sparse.mtx').T
adata.var_names = genes
adata.obs_names = barcodes

# Load and align metadata
metadata = pd.read_csv('metadata.csv', index_col=0)
adata.obs = metadata.loc[adata.obs_names]

# Select number of cells for the subset
n_cells = 6000
n_cells = min(n_cells, adata.n_obs)

# Randomly sample cells
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(adata.n_obs, size=n_cells, replace=False)
adata_subset = adata[random_indices, :]

# Save subset to .h5ad file
adata_subset.write("sample_data.h5ad")
```

The processed_data.h5ad file in this folder was created using the command

```bash
./execution.sh --process command
```


## ⚠️ Notes

The full dataset is not included in this repository due to GitHub file size limits.
This subset is intended for demonstration, testing, and prototyping workflows using scanpy and other single-cell analysis tools.
For full data access, please visit NCBI GEO: GSE176078.

## 🔧 Requirements

To run the script, make sure the following Python packages are installed:

- scanpy
- pandas
- scipy
- numpy

---

For questions or issues, feel free to open an issue or contact the repository maintainer.
