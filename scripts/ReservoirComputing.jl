using Pkg
Pkg.activate(".")

using OpenML, MLJ, DataFrames, CSV, MLJBase, ReservoirComputing, ParameterizedFunctions, OrdinaryDiffEq


begin
    train_x = CSV.read(joinpath("data/train_nf_2_x.csv"), DataFrame)
    train_y = (CSV.read(joinpath("data", "training_nf.csv"), DataFrame)).precipitation_nextday
    test = CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame)
end



begin
    approx_res_size = 300
    radius = 1.2
    degree = 6
    activation = tanh
    sigma = 0.1
    beta = 0.0
    alpha = 1.0
    extended_states = false
end;

begin
    esn = ESN(approx_res_size,
        train_x,
        degree,
        radius,
        activation = activation, #default = tanh
        alpha = alpha, #default = 1.0
        sigma = sigma) #default = 0.1
end

