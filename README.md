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
2025. DOI: TBA.

The preprint is available at TBA.

<!-- GETTING STARTED -->
## Getting Started

We provide a simple example for training and testing a neural network for 
parameter extraction of 2D traininstors in the training_example_2D_FET 
directory. We also provide an example for high-electron-mobility transistors
in the training_example_HEMT directory [not yet added; work in progress].

See the README.md files within these directory for specific usage details.

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
