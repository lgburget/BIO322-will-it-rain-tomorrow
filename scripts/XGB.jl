using Pkg
Pkg.activate(".")

using MLJ, MLJXGBoostInterface, DataFrames, MLJDecisionTreeInterface, Plots, CSV

features = CSV.read(joinpath("data", "selected_features.csv"), DataFrame)
features_2 = CSV.read(joinpath("data", "selected_features_nf.csv"), DataFrame)
train = CSV.read(joinpath("data", "training_data_filled.csv"), DataFrame)
test = CSV.read(joinpath("data", "test_data_filled.csv"), DataFrame)

train_nf = CSV.read(joinpath("data", "training_nf.csv"), DataFrame)
test_nf = CSV.read(joinpath("data", "test_nf.csv"), DataFrame)

#removing the ":" before the name of the feature and selecting the corresponding 
#columns in the test dataset 


function transforming_data(f, tes, tra)
    s = copy(f)
    
    for i in 1:length(f.features)
        s.features[i] = f.features[i][2:end]
    end
    println(s)
    tr = tra[:,s.features]
    te = tes[:,s.features]
    
    return te, tr
end


#CSV.write("data/train_x.csv", train_x)
#CSV.write("data/train_y.csv", train_y)
#CSV.write("data/test.csv", t)
 

begin
    xgb = XGBoostClassifier(max_depth = 3)
    mod = TunedModel(model = xgb,
                    resampling = CV(nfolds = 5),
                    tuning = Grid(goal = 256),
                    range = [range(xgb, :eta, lower = 0.035, upper = 0.055),
                                range(xgb, :num_round, lower = 200, upper = 500)])
                                     
    m = machine(mod, train_x, train_y.precipitation_nextday)
    fit!(m)
end


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


#testing the good shape of the submission
#sub = CSV.read(joinpath("submissions", "submission_2_xgb_selected_features.csv"), DataFrame)


#saving the machine
function saving_machine(name, machine)
    filename = string("machines/", name, ".jlso")
    MLJ.save(filename, machine)
end



##########################################################################################



#We're now going to fit a xgb model on all features + the new features


train_nf_x = select(train_nf, Not(:precipitation_nextday))
train_nf_y = select(coerce(train_nf, :precipitation_nextday => Multiclass), :precipitation_nextday)

begin
    xgb = XGBoostClassifier(max_depth = 3)
    mod_nf = TunedModel(model = xgb,
                    resampling = CV(nfolds = 7),
                    tuning = Grid(goal = 9),
                    range = [range(xgb, :eta, lower = 0.045, upper = .06),
                                range(xgb, :num_round, lower = 350, upper = 450),])
                                     
    m_nf = machine(mod_nf, train_nf_x, train_nf_y.precipitation_nextday)
    fit!(m_nf)
end 

ev1 = evaluate!(machine(report(m_nf).best_model, train_nf_x, train_nf_y.precipitation_nextday), measure = AreaUnderCurve())
ev1.measurement

saving_machine("5_xgb_all_nf", m_nf)
saving_predictions("5_xgb_all_nf", m_nf, train_nf_x, test_nf)

sub = CSV.read(joinpath("submissions", "submission_5_xgb_all_nf.csv"), DataFrame)

#####################################################################################################


#introducing the reduced dataset with limited features including the artificialy created ones

train_nf_2_x, test_nf_2 = transforming_data(features_2, train_nf_x, test_nf)

test_nf_2
train_nf_2_x

CSV.write("data/test_nf_2.csv", test_nf_2)
CSV.write("data/train_nf_2_x.csv", train_nf_2_x)


begin
    xgb = XGBoostClassifier(max_depth = 3)
    mod_nf_2 = TunedModel(model = xgb,
                    resampling = CV(nfolds = 7),
                    tuning = Grid(goal = 9),
                    range = [range(xgb, :eta, lower = 0.06, upper = 0.08),
                                range(xgb, :num_round, lower = 350, upper = 450),])
                                     
    m_nf_2 = machine(mod_nf_2, train_nf_2_x, train_nf_y.precipitation_nextday)
    fit!(m_nf_2)
end 

ev2 = evaluate!(machine(report(m_nf_2).best_model, train_nf_2_x, train_nf_y.precipitation_nextday), measure = AreaUnderCurve())
ev2.measurement

saving_machine("7_xgb_nf_2", m_nf_2)
saving_predictions("7_xgb_nf_2", m_nf_2, train_nf_2_x, test_nf_2)