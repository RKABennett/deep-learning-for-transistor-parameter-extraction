## About The Project

This repository contains the code and results for a deep learning approach for
transistor parameter extraction. All code is implemented in Python; all data for
training is available in both the original Sentaurus csv output and compiled 
into useful NumPy arrays; and all neural networks are implemented in TensorFlow.

For full details and results, please see our paper, available open access at 
[https://spj.science.org/doi/10.34133/research.1103](https://spj.science.org/doi/10.34133/research.1103).

## Updates
* 2026 January 01 -- The in-press version of our paper has been published in *Research*! You can view it at [https://spj.science.org/doi/10.34133/research.1103](https://spj.science.org/doi/10.34133/research.1103).
* 2025 December 29 -- Our paper has been accepted for publication in the Science Partner Journal *Research*! 🎉🎉🎉
* 2025 July 07 -- Our GitHub is live!
* 2025 July 07 -- Our [arXiv preprint](https://arxiv.org/abs/2507.05134) is live!

## Installation

To create and activate a virtual environment:  

```bash
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

To install the required dependencies, do:  

```bash
pip install -r requirements.txt
```

Afterwards, to install this package (required to run the scripts in 
[the demo directory](./demo) and 
[the demo notebook](./../demo_notebook/demo_notebook.ipynb)): 

```bash
pip install -e .
```

Note that this will allow tensorflow to run in GPU mode if you have a GPU; 
otherwise, it will run on your CPU.

<!-- REPOSITORY LAYOUT -->
## Repository layout
Key files and directories of this project are:

- [config.json](./config.json) -- A config file where key variables are defined.
- [data](./data)    -- Sentaurus simulation data from our preprint.
- [demo](./demo)    -- A training example using data from our preprint.
- [demo_notebook](./demo_notebook)    -- A training notebook example using data from our preprint.
- [models](./models)  -- Sample pretrained models.
- [src](./src)     -- Core code for this project.


<!-- GETTING STARTED -->
## Getting Started

We provide a simple example for training and testing a neural network for 
parameter extraction of 2D transistors in the [demo directory](./demo). We
also provide a walkthrough notebook in the 
[demo_notebook directory](./demo_notebook).

See the [README file in the demo directory](./demo/README.md) for specific 
usage details.

<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](./LICENSE).

<!-- CITING THIS WORK-->
## Citing this work
If you use this code or find our project helpful, please cite our in-press paper at [https://spj.science.org/doi/10.34133/research.1103](https://spj.science.org/doi/10.34133/research.1103):

R.K.A. Bennett, J.L. Uslu, H.F. Gault, L. Hoang, A.I. Khan, T. Pena,
K. Neilson, Y.S. Song, Z. Zhang, A.J. Mannix, E. Pop, "Deep Learning to Automate 
Parameter Extraction and Model Fitting of Two-Dimensional Transistors," Research,
2026. doi:10.34133/research.1103.

```bibtex
@article{Bennett2026DeepLearning,
  title        = {Deep Learning to Automate Parameter Extraction and Model 
                  Fitting of Two-Dimensional Transistors},
  author       = {Bennett, R. K. A. and Uslu, J. L. and Gault, H. F. and 
                  Hoang, L. and Khan, A. I. and Pena, T. and Neilson, K. and 
                  Song, Y. S. and Zhang, Z. and Mannix, A. J. and Pop, E.},
  journal      = {Research},
  year         = {2026},
  doi          = {10.34133/research.1103},
  url          = {https://spj.science.org/doi/10.34133/research.1103}
}
```

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* Funding sources: NSERC, SRC SUPREME Center, SystemX, Stanford Graduate 
Fellowship Program

<!-- CONTACT -->

## Contact

Issues, questions, comments, or concerns? Please email Rob at 
rkabenne [at] stanford [dot] edu.
