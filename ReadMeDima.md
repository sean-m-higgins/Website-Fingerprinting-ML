# Website-Fingerprinting-ML
Machine Learning implementation of website fingerprinting


## Interested in running this website fingerprinting program?
Follow along with the steps below to get started


## Python files
The important python files are located in /tests/train_tests
- one_website.py : takes one hardcoded website (facebook.com) and training data is built consisting of half facebook.com examples and the other half are random examples from random websites in our data not including any facebook.com. 
- all_websites.py : 
- model_selection.py : first half of file is a copy of one_website.py and then a grid search is run for each of the four classifiers with both sets of data. At the end an n-fold cross validation is run on the best model.


## Step 1: Setting up datasets
Clone this repository as is, but download the datasets yourself and drag into data folder

Make sure to expand the datasets into the data folder so that you see three folders labeled
- 10s_website1
- 20s_website1
- result

These all will be ignored for a git push in case we want to leave guidelines for any specific things to do within that folder


## Step 2: Running the code
From the home directory /Website-Fingerprinting-ML run: 
- `python /tests/train_tests/one_website.py`
- `python /tests/train_tests/all_websites.py`
- `python /tests/train_tests/model_selection.py`
