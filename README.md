# Final Project for the Computer Science Foundation Class
---
## Description:

#### This Final Project is for our Computer Science Foundation class.
#### We have a group of 2 data scientists: Yupeng Yang and Deen Huang. We build this publicly available repository on GitHub and our project team uses git for version control.
---
### Components of our Repository:

* README.md

* AUTHORS.md

* LICENSE.md

* requirements.txt

* .gitignore

* setup.py

* Package folder: csf_project_folder:

> - config.py: the config info part, including data paths and model paths.

> - data_process.py: this part includes data loading and preprocessing.

> - model.py: this part includes machine learning models we used from sklearn.

> - main_train.py: this part describes how we train the model, steps includes load train data, model train, and validate model.

> - main_predict.py: this part describes how we predict by the model, steps includes load data, load model, segment words and model predict.

### Installation
Type the following commands in your terminal.
```
git clone https://github.com/deenhuang/CSF-DATS-6450-FINAL.git

cd CSF-DATS-6450-FINAL

python3 setup.py install 
```
