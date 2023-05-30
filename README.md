# Comparing-Machine-Learning-based-Methods-MPC-BOPTEST

This repository implements the scientific paper "Comparing Machine Learning based Methods MPC BOPTEST". The purpose of this project is to provide a concrete and comprehensive comparison of machine learning methods in the context of Model Predictive Control (MPC) using the BOPTEST HVAC simulator.

## Disclaimer
<font color="orange">

Please note that while this repository implements the methodology outlined in the paper "Comparing Machine Learning based Methods MPC BOPTEST", I am not one of the original authors of that paper. This implementation is meant to be a resource for further research and development in this field, and all credit for the original methodology and ideas goes to the original authors of the paper.
</font>

## Repository Structure

This repository is organized as follows:
```
├── 0_generate_dataset.py # Script for generating the dataset
├── 1_train_model.py # Script for training the model
├── 2_eval_model.py # Script for evaluating the model
├── 3_make_experiment.py # Script for running experiments
├── boptest.py # Main BOPTEST interface
├── _cache_models # Cached models for quick reloading
├── controller # Control logic and algorithms
├── dataset # Data for training and testing
├── env.py # Environment setup and variables
├── LICENSE # License details
├── models # Models and Model architecture ressources
├── project1_boptest # Project-specific BOPTEST configurations
├── pycache # Cached Python bytecode
├── README.md 
└── TODO # Planned features and improvements
```

## Installation

1. Clone this repository to your local machine using `https://github.com/Enderdead/Comparing-Machine-Learning-based-Methods-MPC-BOPTEST.git`.
2. Ensure that you have the necessary Python packages installed. (You can list these in a `requirements.txt` file for easy installation with `pip install -r requirements.txt`.)

## Usage

To use this project, follow these steps:

1. Generate the dataset by running `python 0_generate_dataset.py`.
2. Train the model by running `python 1_train_model.py`.
3. Evaluate the model by running `python 2_eval_model.py`.
4. To make an experiment, run `python 3_make_experiment.py`.

## Contributing

We welcome contributions to this project. Please feel free to submit a pull request or open an issue for discussion.

## License

This project is licensed under the terms of the GNU-V3 License. For more details, see the LICENSE file.

## Acknowledgements

We would like to thank the authors of the original "Comparing Machine Learning based Methods MPC BOPTEST" paper for their contributions to the field. 
