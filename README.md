# ML - Labs Overview

This repository contains the laboratory assignments for the course *Machine Learning I*:

- **Lab 1**: Naive Bayes classifier  
- **Lab 2**: k-Nearest Neighbors (kNN) classifier  
- **Lab 3**: Shallow neural networks, cross-validation and model selection  
- **Lab 4**: Basic autoencoder  

Each lab is organized in its own dedicated folder, and the final **report** consolidates the results, methodologies, and insights from all the assignments.

---

## Launch the programs
Before running any lab, you must create a virtual environment and install the required dependencies. 
```
# Create the virtual environment
python3 -m venv .venv
source .venv/bin/activate # (use 'deactivate' to exit)

# Install dependencies
pip install numpy pandas scikit-learn
pip install matplotlib
pip install tensorflow
```
Each lab contains its own ```main.py ```.
Example:
```
cd lab3
python mainTask.py
```


### Troubleshooting
#### 'pip' error (Python 3.12)
If you cannot upgrade pip (common with Python 3.12), recreate the environment as follows:
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
---
