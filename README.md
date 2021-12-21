# BIO322-will-it-rain-tomorrow

This repository contains the code of my project for the machine learning class BIO-322. The goal of this project is to predict rainfall in Pully based on the previous day's meteorological data. The project was set as a competition on [kaggle](https://www.kaggle.com), a machine learning competition platform. All the code here is in the Julia programmation language. 

## Organisation

The repository is organised as follows:
 - The data exploration is presented as a [Pluto](https://www.juliapackages.com/p/pluto) notebook
 - The methods are in the folder scripts and are coded in the Julia language. They can be accessed via any code editor like [VSCode](https://code.visualstudio.com/)
 - The trained machines are in the machines folder
 - The kaggle submissions are in the submissions folder
 - The datasets (original and modified) are in the data folder
 - The predictions on training and test datasets are in the prediction folder

## Setting up
To use the code, please download [julia](https://julialang.org/downloads) (at least version 1.6.2)
open julia and install the code in this repository with
```julia
julia> using Pkg
       Pkg.develop(url = "https://github.com/lgburget/BIO322-will-it-rain-tomorrow")
       Pkg.activate(".")
       Pkg.instantiate()
```
## Data Exploration
To use the data exploration notebook, start julia and type
```julia
julia> using Pkg; Pkg.activate(".")
       using Pluto
       Pluto.run()
```
Once the Pluto homepage opens, load the file by typing "data_visualization.jl" in the "Open from file" space

The data exploration notebook contains:
 - Initial dataset engineering and data visualization
 - Feature engineering
 - A detailed list of submissions with the code of the machine, the optimal parameters found and the score obtained
 - A way to load machines and prediction to visualize the results post-training

## Scripts
To use the scripts, you only need to load them in a code editor supporting the Julia programmation language

Scripts contain:
 - logistic_classifier.jl: an implementation of a standard linear classifier using MLJLinearModels
 - XGB.jl: an implementation of gradient boosting on different engineered datasets using MLJXGBoostInterface
 - neural_network.jl: an implementation of deep neural networks using MLJFlux
 - feature_selection.jl: an implementation of recursive feature elimination with cross validation (RFECV) using scikitlearn
 - utility.jl: a file containing utilitary functions used by the other files


