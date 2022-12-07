Msc thesis in Data Analytics. 2019
==================================

Msc thesis on data analytics, DIT 2019

Project Organization
--------------------

    ├── LICENSE            <- Legal notice details
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a YYY/MM/DD date (for 
    │                         ordering), and a short `-` delimited description, e. g.,  
    │                         `2018/10/20-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Thesis report (LaTeX)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
     ├── __init__.py       <- Makes src a Python module
     │
     ├── data              <- Scripts to download or generate data
     │   └── make_dataset.py
     │
     ├── features          <- Scripts to turn raw data into features for modeling
     │   └── build_features.py
     │
     └── models           <- Scripts to train models and then use trained models to make
      │                      predictions
      ├── predict_model.py
      └── train_model.py

--------

How to install this project
---------------------------

This project needs a number of dependencies to work:

1. Install Anaconda python. See instrucioons for your OS at [project's website](https://www.anaconda.com/).
2. Create python environment using: 
3. Install ffmpeg libraries. See instrucionts for your OS at [project's website](https://www.ffmpeg.org/).


How to reproduce this research
------------------------------

From projects root folder: 

0. Update '''.env''' file with a proper path for the '''path.root''' variable
1. '''make data''' will download and pre-process the dataset.
2. '''make features''' will compute features (MFCC, roll_off ...)
3. '''make train name=MODEL''' will train the indicated model.

Authorship and mentions
-----------------------

Author: Víctor Santiago González
Contact: viktorsantiagogonzalez@gmail.com

<p>
  <small>
    Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience
  </small>
</p>
