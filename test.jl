using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse
using OpenML, MLJ, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJFlux, Flux, MLJDecisionTreeInterface, Plots, CSV, 
StatsPlots, LinearAlgebra, Statistics, MLJMultivariateStatsInterface

training_data = CSV.read("data/trainingdata.csv", DataFrame)

describe(training_data)

begin
    coerce!(training_data, :precipitation_nextday => Multiclass)
    coerce!(training_data, Count => Continuous)

    MLJ.transform()

    training_data_x = select(training_data, Not(:precipitation_nextday))
    training_data_y = training_data.precipitation_nextday
end;

begin
    m1 = machine(LogisticClassifier(), training_data_x, training_data_y) |> fit!

end