using Pkg
Pkg.activate(".")

using DataFrames, Plots, CSV, ScikitLearnBase, ScikitLearn, Conda, Random

Random.seed!(42)
include("utility.jl")


@sk_import ensemble: RandomForestClassifier
@sk_import feature_selection: RFECV


train = CSV.read(joinpath("data/training_data_filled.csv"), DataFrame)
train_x = select(train, Not(:precipitation_nextday))
train_y = train.precipitation_nextday
test = CSV.read(joinpath("data/testdata.csv"), DataFrame)

train_nf = CSV.read(joinpath("data", "training_nf.csv"), DataFrame)
test_nf = CSV.read(joinpath("data", "test_nf.csv"), DataFrame)


using ScikitLearn: fit!, predict

function selecting_features(train)
    n = [string(":", names(train)[i])  for i in 1:length(names(train))-1]
    r =RandomForestClassifier(n_jobs = -1, verbose = 2)

    r.fit(Matrix(select(train, Not(:precipitation_nextday))),train.precipitation_nextday)
    f_i = DataFrame(names = n, score = r.feature_importances_)
    CSV.write("data/feature_importance.csv", f_i)

    rfe = RFECV(r, cv=5, scoring="accuracy", verbose = 2)
    rfe.fit(Matrix(select(train, Not(:precipitation_nextday))),train.precipitation_nextday)
    #took 30min
    
    selected_features = DataFrame(features = n[rfe.get_support()])

    return selected_features
end
s = DataFrame(names = selecting_features(train_nf).features)

CSV.write("data/selected_features_nf.csv", s)
fi = CSV.read(joinpath("data/feature_importance_nf.csv"), DataFrame)
fs = sort!(fi, :score, rev=true)

######################################################################################################
# We can then use the function transform_data from utility.jl to create a training and a test dataset 
# from only including the features that we want
######################################################################################################
