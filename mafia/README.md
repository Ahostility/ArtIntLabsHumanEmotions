# Who is the MAFIA?
![Are you a mafia?](logo.jpg)
# Useful links
## Introduction
* [Art-Int Labs - Dev Convention](https://docs.google.com/document/d/1qs7vi74OvmC4_PsLBfB3ezuobYa_O34mdIJZJHnfdys/edit)
* [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)
* [Project Structure](https://drivendata.github.io/cookiecutter-data-science/)
* [Project Architecture](https://app.diagrams.net/#G1iJksSBIb5lPlgQm7wF63_lALs-z9lZip)
* [Catalyst Ecosystem](https://catalyst-team.github.io/catalyst/)
* [tar + gzip](https://www.howtogeek.com/248780/how-to-compress-and-extract-files-using-the-tar-command-on-linux/)
* [Bash ZIP commands for Windows](http://stahlworks.com/dev/index.php?tool=zipunzip)
* [Mafia Markup (Google Sheets)](https://docs.google.com/spreadsheets/d/1FBzMgIyHwBZjK7LXHRejphz5x802ytXdIyRwWFrUw1A/edit#gid=1227655998)
## Version Control
* [Git Book](https://git-scm.com/book/ru/v2/)
* [DVC Book](https://dvc.org/doc/home)
* [DVC on Habr (tutorial)](https://habr.com/ru/company/raiffeisenbank/blog/461803/)
## PyTorch
* [About `torch.utils.data`](https://pytorch.org/docs/stable/data.html)
* [Model summary (like `model.summary()` in Keras)](https://github.com/sksq96/pytorch-summary)
* [Writing Custom Datasets, DataLoaders and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
* [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)

# Directory structure
```
├── data
│   ├── models         <- Trained and serialized models, model predictions, or model summariessources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── data           <- Scripts to download or generate data and turn raw data into features for modeling
│   │
│   ├── experiments    <- Scripts to experiment models and then use trained models to make
│   │                     predictions
│   │
│   └── modules        <- Cool code
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── LICENSE
└── README.md          <- The top-level README for developers using this project.
```

# Installing the environment
## On PC (through *git clone*)
```bash
git clone -b <branch> https://github.com/pandov/mafia.git
```
### **or**
```bash
git clone https://github.com/pandov/mafia.git
cd mafia
git checkout -b <branch>
```

## On [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true)
**IMPORTANT!** CREATE A COPY OF NOTEBOOK: *File -> Save a copy in...*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pandov/mafia/blob/master/notebooks/colab.ipynb)
# Git help
### Create and push new branch
```bash
git checkout -b <branch>
git push -u origin <branch>
```
### Get the file from another branch:
```bash
git fetch
git checkout origin/<branch> -- <filepath>
```

# DVC help
### How to add your raw dataset?
Just version it using DVC and push:
```bash
dvc add data/raw/<dataset_folder>
dvc push data/raw/<dataset_folder>.dvc
```
Then you can delete the `data/raw/<dataset_folder>` and download it again:
```bash
dvc pull data/raw/<dataset_folder>.dvc
```
