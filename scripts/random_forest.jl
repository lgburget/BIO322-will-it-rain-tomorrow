using Pkg
Pkg.activate(".")

using MLJ, MLJXGBoostInterface, DataFrames, MLJDecisionTreeInterface, Plots, CSV, Random

Random.seed!(42)
include("utility.jl")


# RANDOM FOREST FILE
# The goal of this file is to train and evaluate random tree classifiers on the selected 390 best 
# features, which achieves an AUC of 0.918 on the cross- validation.
# 
# Name of the best machine: 42_xgb_nf_2, its script can be loaded in the data_visualization.jl notebook


begin
	train_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "train_nf_2"), DataFrame))),CSV.read(joinpath("data", "train_nf_2"), DataFrame))
	test_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame))),CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame))

	train_nf_2_x = select(train_nf_2, Not(:precipitation_nextday))
	train_nf_2_y = coerce!(train_nf_2, :precipitation_nextday => Multiclass).precipitation_nextday
end

begin
    model = RandomForestClassifier()
	
    self_tuned_model = TunedModel(model = model,
                            resampling = CV(nfolds = 7),
                            tuning = Grid(goal = 16),
                            range = 
						 [range(model, :n_trees, lower = 100, upper = 1000),
                          range(model, :max_depth, values = [3,4,5,6])])
 
    m = machine(self_tuned_model, train_nf_2_x, train_nf_2_y) |> fit!
end

evaluate!(machine(report(m).best_model, train_nf_2_x, train_nf_2_y) , resampling = CV(rng = 42), measure = AreaUnderCurve())

saving_machine("41_rf_nf_2", m)
saving_predictions("41_rf_nf_2", m, train_nf_2_x, test_nf_2)