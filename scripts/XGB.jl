using Pkg
Pkg.activate(".")

using MLJ, MLJXGBoostInterface, DataFrames, MLJDecisionTreeInterface, Plots, CSV, Random

Random.seed!(42)
include("utility.jl")


# GRADIENT BOOSTING FILE
# The goal of this file is to train and evaluate gradient boosting based classifiers on diffierent 
# training sets. The best performing classifier is the third one (fitted on 390 selected features 
# including the artificially created ones selection), which achieves an AUC of 0.9379 on the 
# cross- validation and a score of 0.95306 on the public leaderboard. In this case the addition 
# of new features and the elimination of less important features did improve the model's accuracy
# 
# Name of the best machine: 26_xgb_nf_2, its script can be found in the data_visualization.jl notebook
#
# measures:
#
# original dataset:                                 |        auc = 0.93416
# best original features:                           |        auc = 0.93516
# best features (including the engineered ones):    |    !!  auc = 0.93791   !!


begin
    train = CSV.read(joinpath("data", "training_data_filled.csv"), DataFrame)
    test = CSV.read(joinpath("data", "test_data_filled.csv"), DataFrame)
    train_x = select(train, Not(:precipitation_nextday))
    train_y = coerce!(train, :precipitation_nextday => Multiclass).precipitation_nextday
end;

begin
    train_nf = CSV.read(joinpath("data", "training_nf.csv"), DataFrame)
    test_nf = CSV.read(joinpath("data", "test_nf.csv"), DataFrame)
    train_nf_x = select(train_nf, Not(:precipitation_nextday))
    train_nf_y = coerce!(train_nf, :precipitation_nextday => Multiclass).precipitation_nextday
end;

begin
    train_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "train_nf_2"), DataFrame))),CSV.read(joinpath("data", "train_nf_2"), DataFrame))
    test_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame))),CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame))
    train_nf_2_x = select(train_nf_2, Not(:precipitation_nextday))
    train_nf_2_y = coerce!(train_nf_2, :precipitation_nextday => Multiclass).precipitation_nextday
end;

##########################################################################################
# best xgb model on the initial dataset (submission 3 in data_visualization.jl)

begin
    xgb = XGBoostClassifier(max_depth = 3)
    mod1 = TunedModel(model = xgb,
                    resampling = CV(nfolds = 5),
                    tuning = Grid(goal = 9),
                    range = [range(xgb, :eta, lower = 0.03, upper = 0.05),
                                range(xgb, :num_round, lower = 480, upper = 520)],
                    measure = auc)
                                     
    m1 = machine(mod1, train_x, train_y)
    fit!(m1)
end

ev1 = evaluate!(machine(report(m1).best_model, train_x, train_y), measure = AreaUnderCurve())
ev1.measurement


##########################################################################################

# We're now going to fit a xgb model on all features + the new features
# best model: submission 1 with selected parameters in data_visualization.jl

begin
    xgb = XGBoostClassifier(max_depth = 3)
    mod_nf = TunedModel(model = xgb,
                    resampling = CV(nfolds = 7),
                    tuning = Grid(goal = 9),
                    range = [range(xgb, :eta, lower = 0.045, upper = .06),
                                range(xgb, :num_round, lower = 350, upper = 450),],
                    measure = auc)
                                     
    m_nf = machine(mod_nf, train_nf_x, train_nf_y)
    fit!(m_nf)
end 

ev2 = evaluate!(machine(report(m_nf).best_model, train_nf_x, train_nf_y), measure = AreaUnderCurve())
ev2.measurement


#####################################################################################################

# We're now going to fit a xgb model on the 321 selected features (including the new features)

begin
    xgb = XGBoostClassifier(max_depth = 3)
    mod_nf_2 = TunedModel(model = xgb,
                    resampling = CV(nfolds = 7),
                    tuning = Grid(goal = 9),
                    range = [range(xgb, :eta, lower = 0.050, upper = .060),
                            range(xgb, :num_round, lower = 450, upper = 650)],
                    measure = auc)
                                     
    m_nf_2 = machine(mod_nf_2, train_nf_2_x, train_nf_2_y)
    fit!(m_nf_2)
end 

ev2 = evaluate!(machine(report(m_nf_2).best_model, train_nf_2_x, train_nf_2_y), measure = AreaUnderCurve())
ev2.measurement


#####################################################################################################

saving_machine("26_xgb_nf_2", m_nf_2)
saving_predictions("26_xgb_nf_2", m_nf_2, train_nf_2_x, test_nf_2)