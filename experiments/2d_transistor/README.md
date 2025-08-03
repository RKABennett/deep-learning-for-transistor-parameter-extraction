<!-- ABOUT THE PROJECT -->
## Training example overview

This directory contains a simple example where we train a neural network to 
predict the model parameters of 2D transistors. We then validate our neural 
network by extracting model parameters for a test set and seeing how well they 
reproduce the original Id-Vgs curves.

If you are confused about the physics we are using, the device setup, or why we 
are doing specific machine learning things, most of your questions will probably 
be answered by our preprint (citation below). If you are still confused, please 
email Rob at rkabenne [at] [stanford] [dot] edu.

## Quick usage

To train the neural network, run these Python files in the following order:

1. `process_data.py`
2. `train_model.py` [This takes ~10 minutes on an NVIDIA RTX 4080 SUPER]

To test the trained forward neural network, run these Python files in the 
following order:

1. `test_forward_0_save-fits.py`
2. `test_forward_1_plots-fits.py`

To test the trained inverse neural network, run these Python files in the 
following order:

1. `test_inverse_0_save-fits.py`
2. `test_forward_1_param-extract.py`
3. `test_forward_2_plot-fits.py`

## Key directories and files

* `data` -- Sentaurus data for our training, dev, and test sets. Each 
  subdirectory within this directory contains Id-Vgs curves for Vds = 0.1 and 
  1 V. It also contains `variables.csv`, which contain the values for each of 
  the eight fitting parameters used in our model.

  Fitting parameters, names, units are:
  
  * mobility (`mobility`) [cm^2 V^-1 s^-1]
  * electron affinity (`chi0`) [eV]
  * effective density of states (`DOSpeak`) [cm^-3]
  * concentration of gaussian-like donors (`DonorConc`) [cm^-2 eV^-1]
  * median energy of donors (`DonorEmid`) [eV]
  * energy width of donors (`DonorEsig`) [eV]
  * peak concentration of exponential acceptor traps (`bandtailconc`) 
    [cm^-3 eV^-1]
  * energy width of exponential acceptor traps (`bandtailwidth`) [eV]

* `NN_fns.py` -- Library with many useful functions for processing data, and 
  training/testing our neural networks.

* `NN_variables.py` -- Where we input settings for data processing and training 
  our networks. Keeping things in one file can make training multiple networks 
  consistent, since it gives us one place to keep all of our variables.

* `process_data.py` -- Python script that preprocesses our raw data by 
  implementing the interpolation, noise floor, and scaling schemes described in 
  the supplementary information of our preprint. After, it saves numpy arrays of 
  our input data (Id-Vgs curves and engineered features) and output data 
  (physical model parameters).

* `train_model.py` -- Python script that implements the full training procedure, 
  including:  
  - Training the forward neural network that we use in the tandem solver  
  - Building the augmented dataset  
  - Pre-training the model on the augmented dataset  
  - Fine-tuning the model on the Sentaurus dataset

* `test_forward_0_save-fits.py` -- Runs the test set through the forward model, 
  saving the resultant fits to get them ready to plot. Also saves a file 
  containing all of our R^2 values so that we can analyze our errors.

* `test_forward_1_plot-fits.py` -- Plots the fits generated above.

* `test_inverse_0_save-fits.py` -- Runs the test set through the inverse NN and 
  extracts the model parameters for every device in the test set. It then 
  imports the model parameters into a very well trained forward solver that 
  approximates Sentaurus extremely well (see the end of Supplementary Section 4 
  in our preprint) and estimates the currents that are associated with these 
  model parameters. We save the currents to file to make them easier to plot.

* `test_inverse_1_param-extract.py` -- Runs the test set through the inverse NN 
  and extracts and plots the actual vs. predicted model parameters.

* `test_inverse_2_plot-fits.py` -- Plots the fits generated above.

* `NN_forward_well_trained.keras` -- Neural network (Tensorflow, Keras) model
that is trained to approximate Sentaurus for our 2D devices. Note that this 
model is ONLY valid for 2D devices with the specific device geometries, 
model parameters, and scaling as implemented in this folder. If you change any
of these, then this network will output garbage.

