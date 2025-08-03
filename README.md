<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains the code and results for a deep learning approach for
transistor parameter extraction. All code is implemented in Python; all data for
training is available in both the original Sentaurus csv output and compiled 
into useful NumPy arrays; and all neural networks are implemented in TensorFlow.

For full details and results, please see the following prepreint:

R.K.A. Bennett, J.L. Uslu, H.F. Gault, L. Hoang, A.I. Khan, L. Hoang, T. Pena,
K. Neilson, Y.S. Song, Z. Zhang, A.J. Mannix, E. Pop, "Deep Learning to Automate 
Parameter Extraction and Model Fitting of Two-Dimensional Transistors," arXiv,
2025. doi:10.48550/arXiv.2507.05134.

The preprint is available at https://arxiv.org/abs/2507.05134

<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-learning-for-transistor-parameter-extraction.git
cd deep-learning-for-transistor-parameter-extraction
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate transistor-param-extraction
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Project Structure

```
├── src/                          # Source code
│   └── transistor_param_extraction/
│       ├── NN_fns.py            # Neural network functions
│       ├── NN_variables.py      # Configuration variables
│       └── cli.py               # Command line interface
├── experiments/                  # Experimental configurations
│   └── 2d_transistor/           # 2D transistor experiment
├── data/                        # Data storage
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed datasets
├── models/                      # Trained models
│   └── trained/                 # Production models
├── results/                     # Experimental results
│   ├── plots/                   # Generated plots
│   └── metrics/                 # Performance metrics
├── configs/                     # Configuration files
├── tests/                       # Unit tests
└── notebooks/                   # Jupyter notebooks
```

### Quick Start

1. **Train the models:**
```bash
transistor-train --experiment 2d_transistor
```

2. **Test the forward model:**
```bash
transistor-test --model-type forward --experiment 2d_transistor
```

3. **Test the inverse model:**
```bash
transistor-test --model-type inverse --experiment 2d_transistor
```

### Manual Training (Alternative)

Navigate to the experiment directory and run scripts in order:

```bash
cd experiments/2d_transistor
python process_data.py
python train_model.py
python test_forward_0_save-fits.py
python test_forward_1_plot-fits.py
```

<!-- REQUIREMENTS -->
### Requirements

matplotlib==3.6.3

numpy==2.3.1

pandas==2.1.4+dfsg

scikit_learn==1.4.1.post1

scipy==1.16.0

tensorflow==2.17.1

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/) - for this README 
template
* Funding sources: NSERC, SRC SUPREME Center, SystemX, Stanford Graduate 
Fellowship Program

<!-- CONTACT -->
## Contact

Issues, questions, comments, or concerns? Please email Rob at 
rkabenne [at] stanford [dot] edu
