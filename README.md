# Multi-Armed-Bandit

Project for the exam of Probabilistic Machine Learning.

The aim of the project is to use multi-armed-bandit in two different settings: one more theoretical and one more practical.

## Theoretical Section

We implemented bayesian optimization using gaussian processes and MAB. 
The goal is to find the meximum of an objective function which is expensive to evaluate.
Different algorithms are tested.

The files for this part are:

* `GP_MAP.ipynb`
* `gp_algorithms.py` containing the definitions of the acquisition functions
* `gp_utilities.py` containing functions for plotting and updating GP

## Practical Section

We used MAB to select from a pool of advertisements the one which fits best users' preferences.
The dataset is randomly generated to create a group of users and a pool of ads.
One ad will have the highest click-through-ratio (CTR) and an efficient implementation should target it.

The files for this part are:

* `MAP.ipynb`
* `algorithms.py` containing different algorithms for targeting best ad
* `utilities.py` containing various function for plots and dataset 
* `ad.py` and `user.py` containing ad and user classes
