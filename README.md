# EDA - Amphibians Classification - *by Jan Kořínek*
Hello and welcome. 

In this repository you can find exploratory data analysis with trained and mutually compared models for a classification task on the Amphibians dataset.

Complete EDA with model training can be run by`EDA-Amphibians_Classification.ipynb`. Project contains library of the *Python* scripts stored in *lib* folder which are responsible for the data wrangling, model training and evaluation and for plotting of various characteristics.

## Project Structure
    .
    ├── data                        # Training dataset
    ├── export                      # Sample plots
    ├── lib                         # Code library
        ├── misc_functions.py       # Support functions
        ├── ml_oversampling.py      # Class for multi-label oversampling
        ├── models_training.py      # All models training&evaluation pipelines
        ├── prepare_dataset.py      # Dataset preparation functions
    ├── models                      # Trained models storage
    LICENSE
    README.md
    requirements.txt
    EDA-Amphibians_Classification.ipynb     # Main analysis and models training file

## Usage
* Clone the repository below:

`$ git clone https://gitfront.io/r/korinek-j/67a162cc96f0b8781a7e0f36d989e639a110ce22/04-Classification-Amphibians.git`

`$ cd 04_Classification-Amphibians`

* Setup virtual environment in Anaconda, Pycharm or in IDE you're currently using.

* Install libraries in `requirements.txt`

* Run `$ jupyter-lab EDA-Amphibians_Classification.ipynb`