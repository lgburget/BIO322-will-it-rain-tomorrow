using Pkg
Pkg.activate(".")

using MLJ, DataFrames, MLJLinearModels, Plots, CSV, StatsPlots, Statistics, OpenML

train = CSV.read(joinpath("data", "training_data_filled.csv"), DataFrame)
test = CSV.read(joinpath("data", "test_data_filled.csv"), DataFrame)
train_selected_x = CSV.read(joinpath("data", "train_x.csv"), DataFrame)

########################################################################################################
#
#
#
#
########################################################################################################
# UTILITARIES

function saving_predictions(name, machine, train, test)
    prediction_train = predict(machine, train)
    p_train = DataFrame(p = broadcast(pdf, prediction_train, true))
    file_name_train = string("predictions/prediction_train_", name, ".csv")
    CSV.write(file_name_train, p_train)

    prediction_test = predict(machine, test)
    p_test = DataFrame(id = 1:1200, precipitation_nextday = broadcast(pdf, prediction_test, true))
    file_name_test = string("submissions/submission_", name, ".csv")
    CSV.write(file_name_test, p_test)
end

function saving_machine(name, machine)
    filename = string("machines/", name, ".jlso")
    MLJ.save(filename, machine)
end



########################################################################################################
#
#
#
#
########################################################################################################
# Fitting a linear classifier on all our dataset


train_x = select(train, Not(:precipitation_nextday))[1:2500,:]
valid_x = select(train, Not(:precipitation_nextday))[2501:end,:]
train_y = select(coerce(train[1:2500,:], :precipitation_nextday => Multiclass), :precipitation_nextday)
valid_y = select(coerce(train[2501:end,:], :precipitation_nextday => Multiclass), :precipitation_nextday)

begin
    class = MLJLinearModels.LogisticClassifier(penalty = :none )
    mod = TunedModel(model = class,
                    resampling = CV(nfolds = 2),
                    tuning = Grid(goal = 10),
                    range = [range(class, :gamma, lower = 0, upper = 1),
                            range(class, :lambda, lower = 0, upper = 1),
                            range(class, :fit_intercept, values = [:true, :false])])
                                     
    m1 = machine(mod, train_x, train_y.precipitation_nextday)
    MLJ.fit!(m1)
end

ev1 = evaluate!(machine(report(m1).best_model, train_x, train_y.precipitation_nextday), measure = AreaUnderCurve())
ev1.measurement
ev2 = evaluate!(machine(report(m1).best_model, valid_x, valid_y.precipitation_nextday), measure = AreaUnderCurve())
ev2.measurement 

########################################################################################################
#
#
#
#
########################################################################################################
#fitting a linear classifier on the 111 selected features with RFECV (see @feature_selection.jl)

train_sel_x = train_selected_x[1:2500,:]
valid_sel_x = train_selected_x[2501:end,:]

begin
    class = MLJLinearModels.LogisticClassifier(penalty = :none )
    mod = TunedModel(model = class,
                    resampling = CV(nfolds = 2),
                    tuning = Grid(goal = 10),
                    range = [range(class, :gamma, lower = 0, upper = 1),
                            range(class, :lambda, lower = 0, upper = 1),
                            range(class, :fit_intercept, values = [:true, :false])])
                                     
    m2 = machine(mod, train_sel_x, train_y.precipitation_nextday)
    MLJ.fit!(m2)
end

ev3 = evaluate!(machine(report(m2).best_model, train_sel_x, train_y.precipitation_nextday), measure = AreaUnderCurve())
ev3.measurement
ev4 = evaluate!(machine(report(m2).best_model, valid_sel_x, valid_y.precipitation_nextday), measure = AreaUnderCurve())
ev4.measurement 

########################################################################################################
#
#
#
#
########################################################################################################

begin
    class = MLJLinearModels.LogisticClassifier(penalty = :none )
    mod = TunedModel(model = class,
                    resampling = CV(nfolds = 5),
                    tuning = Grid(goal = 10),
                    range = [range(class, :gamma, lower = 0, upper = 1),
                            range(class, :lambda, lower = 0, upper = 1),
                            range(class, :fit_intercept, values = [:true, :false])])
                                     
    m3 = machine(mod, train_sel_x[:,1:50], train_y.precipitation_nextday)
    MLJ.fit!(m3)
end

ev5 = evaluate!(machine(report(m3).best_model, train_sel_x[:,1:50], train_y.precipitation_nextday), measure = AreaUnderCurve())
ev5.measurement
ev6 = evaluate!(machine(report(m3).best_model, valid_sel_x[:,1:50], valid_y.precipitation_nextday), measure = AreaUnderCurve())
ev6.measurement 


########################################################################################################
#
#
#
#
########################################################################################################

begin
    train_nf_2_x = CSV.read(joinpath("data/train_nf_2_x.csv"), DataFrame)[1:2500,:]
    train_nf_2_y = select(coerce( (CSV.read(joinpath("data", "training_nf.csv") , DataFrame)), :precipitation_nextday => Multiclass), :precipitation_nextday)
    test_nf_2 = CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame)
end

begin
    class = MLJLinearModels.LogisticClassifier(penalty = :none )
    mod4 = TunedModel(model = class,
                    resampling = CV(nfolds = 2),
                    tuning = Grid(goal = 2),
                    range = [range(class, :gamma, lower = 0, upper = 1)])
                                     
    m4 = machine(mod4, train_nf_2_x, train_nf_2_y.precipitation_nextday)
    MLJ.fit!(m4)
end

ev7 = evaluate!(machine(report(m4).best_model, train_nf_2_x, train_nf_2_y.precipitation_nextday), measure = AreaUnderCurve(), resampling = CV(nfolds=2))
ev7.measurement

saving_predictions("8_logistic_nf_2", m4, train_nf_2_x, test_nf_2)
saving_machine("8_logistic_nf_2", m4)