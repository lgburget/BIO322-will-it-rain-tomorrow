using Pkg
Pkg.activate(".")

using MLJ, DataFrames, MLJLinearModels, Plots, CSV, StatsPlots, Statistics, OpenML, Random

Random.seed!(42)

# LINEAR CLASSIFIER FILE
# The goal of this file is to train and evaluate basic linear classifiers on diffierent training sets
# The best performing classifier is the second one (fitted on the first feature selection), which 
# achieves an AUC of 0.915  on the cross- validation and a score of 0.94012 on the public leaderboard.
# In this case the addition of new features didn't improve the model's accuracy

train = CSV.read(joinpath("data", "training_data_filled.csv"), DataFrame)
test = CSV.read(joinpath("data", "test_data_filled.csv"), DataFrame)
train_selected_x = CSV.read(joinpath("data", "train_x.csv"), DataFrame)
test_selected = CSV.read(joinpath("data", "test.csv"), DataFrame)


########################################################################################################
#
#
#
#
########################################################################################################
# Fitting a linear classifier on all our dataset


train_x = select(train, Not(:precipitation_nextday))
train_y = select(coerce(train, :precipitation_nextday => Multiclass), :precipitation_nextday)

begin
    class = MLJLinearModels.LogisticClassifier(penalty = :none )
    mod = TunedModel(model = class,
                    resampling = CV(nfolds = 10),
                    tuning = Grid(goal = 2),
                    range = [range(class, :fit_intercept, values = [:true, :false])],
                    measure = auc)
                                     
    m1 = machine(mod, train_x, train_y.precipitation_nextday)
    MLJ.fit!(m1)
end

ev1 = evaluate!(machine(report(m1).best_model, train_x, train_y.precipitation_nextday), measure = AreaUnderCurve())
ev1.measurement

# auc = 0.901

########################################################################################################
#
#
#
#
########################################################################################################
#fitting a linear classifier on the 111 selected features with RFECV (see @feature_selection.jl)

train_sel_x = train_selected_x

begin
    class = MLJLinearModels.LogisticClassifier(penalty = :l2)
    mod = TunedModel(model = class,
                    resampling = CV(nfolds = 10),
                    tuning = Grid(goal = 50),
                    range = [range(class, :lambda, lower = 0, upper = 100)],
                    measure = auc)
                                     
    m2 = machine(mod, train_sel_x, train_y.precipitation_nextday)
    MLJ.fit!(m2)
end

ev2 = evaluate!(machine(report(m2).best_model, train_sel_x, train_y.precipitation_nextday), measure = AreaUnderCurve())
ev2.measurement
report(m2).best_model

#auc = 0.915

########################################################################################################
#
#
#
#
########################################################################################################
# fitting a linear classifier on our 321 selected features with RFECV (including some of the new features)
begin
    train_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "train_nf_2"), DataFrame))),CSV.read(joinpath("data", "train_nf_2"), DataFrame))
    train_nf_x = select(train_nf_2, Not(:precipitation_nextday))
    train_nf_y = coerce!(train_nf_2, :precipitation_nextday => Multiclass).precipitation_nextday
    test_nf_2 = CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame)
end

begin
    class = MLJLinearModels.LogisticClassifier(penalty = :none )
    mod = TunedModel(model = class,
                    resampling = CV(nfolds = 10),
                    tuning = Grid(goal = 2),
                    range = [range(class, :fit_intercept, values = [:true, :false])],
                    measure = auc)
                                     
    m4 = machine(mod4, train_nf_2_x, train_nf_2_y.precipitation_nextday)
    MLJ.fit!(m4)
end

ev7 = evaluate!(machine(report(m4).best_model, train_nf_2_x, train_nf_2_y.precipitation_nextday), measure = AreaUnderCurve(), resampling = CV(nfolds=2))
ev7.measurement

# auc = 0.891

saving_predictions("24_logistic_sel", m2, train_selected_x, test_selected)
saving_machine("24_logistic_sel", m2)