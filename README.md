## About The Project

This repository contains the code and results for a deep learning approach for
transistor parameter extraction. All code is implemented in Python; all data for
training is available in both the original Sentaurus csv output and compiled 
into useful NumPy arrays; and all neural networks are implemented in TensorFlow.

For full details and results, please see the preprint, available at 
[https://arxiv.org/abs/2507.05134](https://arxiv.org/abs/2507.05134).

## Updates
2025 July 07 -- Our GitHub is live!
2025 July 07 -- Our [arXiv preprint](https://arxiv.org/abs/2507.05134) is live!

## Installation

To install the required dependencies, do:  

```bash
pip install -r requirements.txt
```

To install this package: 

```bash
pip install -e .
```
To create and activate a virtual environment:  

```bash
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

<!-- REPOSITORY LAYOUT -->
## Repository layout
Key files directories of this project are:

- [config.json](./config.json) -- A config file where key variables are defined.
- [data](./data)    -- Sentaurus simulation data from our preprint.
- [demo](./demo)    -- A training example using data from our preprint.
- [models](./models)  -- Sample pretrained models.
- [src](./src)     -- Core code for this project.


<!-- GETTING STARTED -->
## Getting Started

We provide a simple example for training and testing a neural network for 
parameter extraction of 2D trainsistors in the demo directory. 

See the [README file in the demo directory](./demo/README.md) for specific 
usage details.

<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](./LICENSE).

<!-- CITING THIS WORK-->
## Citing this work
If you use this code or find our project helpful, please cite [our preprint:](
https://arxiv.org/abs/2507.05134).

R.K.A. Bennett, J.L. Uslu, H.F. Gault, L. Hoang, A.I. Khan, L. Hoang, T. Pena,
K. Neilson, Y.S. Song, Z. Zhang, A.J. Mannix, E. Pop, "Deep Learning to Automate 
Parameter Extraction and Model Fitting of Two-Dimensional Transistors," arXiv,
2025. doi:10.48550/arXiv.2507.05134.

@article{Bennett2025DeepLearning,
  title        = {Deep Learning to Automate Parameter Extraction and Model 
                  Fitting of Two-Dimensional Transistors},
  author       = {Bennett, R. K. A. and Uslu, J. L. and Gault, H. F. and 
                  Hoang, L. and Khan, A. I. and Pena, T. and Neilson, K. and 
                  Song, Y. S. and Zhang, Z. and Mannix, A. J. and Pop, E.},
  journal      = {arXiv preprint arXiv:2507.05134},
  year         = {2025},
  doi          = {10.48550/arXiv.2507.05134},
  url          = {https://arxiv.org/abs/2507.05134}
}

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* Funding sources: NSERC, SRC SUPREME Center, SystemX, Stanford Graduate 
Fellowship Program

<!-- CONTACT -->

## Contact

Issues, questions, comments, or concerns? Please email Rob at 
rkabenne [at] stanford [dot] edu.
