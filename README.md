# ML - Labs Overview

This repository contains the laboratory assignments for the course *Machine Learning I*:

- **Lab 1**: Naive Bayes classifier  
- **Lab 2**: k-Nearest Neighbors (kNN) classifier  
- **Lab 3**: Shallow neural networks, cross-validation and model selection  
- **Lab 4**: Basic autoencoder  

Each lab is organized in its own dedicated folder, and the final **report** consolidates the results, methodologies, and insights from all the assignments.

---

## Launch the programs
Before launch each lab it is necessary install the virtual environment to install the libraries dependency.
```
#create the virtual envieronment
python3 -m venv .venv
source .venv/bin/activate (or deactivate)

# install the dependency
pip install numpy pandas scikit-learn    #Lab1
pip install matplotlib #Lab1 Lab2
pip install tensorflow #Lab2
pip install scikit-learn #Lab2

```

### Troubleshooting
#### 'pip' error
If you don't have 'pip' upgradable, especially if you have the python version **Python 3.12**, you can run this code:
```
# recreate the virtual environment
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate

# update pip
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```
