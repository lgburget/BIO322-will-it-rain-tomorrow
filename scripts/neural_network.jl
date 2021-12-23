using Pkg
Pkg.activate(".")

using DataFrames, MLJ, Plots, MLJFlux, Flux, CSV, Random

Random.seed!(42)

include("utility.jl")

# NEURAL NETWORK FILE
# The goal of this file is to train and evaluate neural network classifiers on the training set 
# including the 390 selected features (including the artificially created ones selection), which 
# achieves an AUC of 0.928 on the cross- validation and a score of 0.95956 on the public leaderboard. 
# Name of the machine: 27_xgb_nf_2, its script can be loaded in the data_visualization.jl notebook
#
# Below that I propose a linear classification using a Neural Network. It achieved an AUC of 0.918 
# when evaluated on the same dataset as the non-linear network and a score of 0.94259 on, the public
# leaderboard.
# Name of the machine: 40_nn_lin_nf_2, its script can be found in the data_visualization.jl notebook


begin
	train_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "train_nf_2"), DataFrame))),CSV.read(joinpath("data", "train_nf_2"), DataFrame))
	test_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame))),CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame))

	train_nf_2_x = select(train_nf_2, Not(:precipitation_nextday))
	train_nf_2_y = coerce!(train_nf_2, :precipitation_nextday => Multiclass).precipitation_nextday
end

n_in = length(names(train_nf_2_x))
n_out = 2

########################################################################################################
#
#
#
#
########################################################################################################
# Non-linear utilization of the neural network

begin
	model_nn = NeuralNetworkClassifier(
							builder =  MLJFlux.@builder(Chain(Dense(n_in, 256, relu),
							 Dense(256,128,relu),
							 Dense(128, n_out))),
                             batch_size = n_in,
                             optimiser = ADAM(),
							 rng = 42) 
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 3),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 29, upper = 31)],
                              measure = auc)
	mach2 = fit!(machine(tuned_model,train_nf_2_x, train_nf_2_y))
end

evaluate!(machine(report(mach2).best_model, train_nf_2_x, train_nf_2_y), resampling = CV(rng = 42), measure = AreaUnderCurve())

saving_machine("39_nn_nf_2", mach2)
saving_predictions("39_nn_nf_2", mach2, train_nf_2_x, test_nf_2)

########################################################################################################
#
#
#
#
########################################################################################################


# Neural network classifier as a linear method

begin
	model_nn_2 = NeuralNetworkClassifier(
							builder =  MLJFlux.@builder(Chain(Dense(n_in, n_out))),
                            batch_size = n_in,
                            optimiser = ADAM()) 

	tuned_model_2 = TunedModel(model = model_nn_2,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 50),
	                          range = [range(model_nn_2,
									     :(epochs),
									     lower = 50, upper = 250)],
                              measure = auc)

	mach3 = fit!(machine(tuned_model_2, train_nf_2_x, train_nf_2_y))
end
evaluate!(mach3, resampling = CV(nfolds = 4, rng = 42), measure = AreaUnderCurve()) 

saving_machine("40_nn_lin_nf_2", mach3)
saving_predictions("40_nn_lin_nf_2", mach3, train_nf_2_x, test_nf_2)