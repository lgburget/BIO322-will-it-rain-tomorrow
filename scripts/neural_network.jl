using Pkg
Pkg.activate(".")

using DataFrames, MLJ, Plots, MLJFlux, Flux, CSV

train_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "train_nf_2"), DataFrame))),CSV.read(joinpath("data", "train_nf_2"), DataFrame))
test_nf_2 = MLJ.transform(fit!(machine(Standardizer(), CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame))),CSV.read(joinpath("data", "test_nf_2.csv"), DataFrame))

train_nf_x = select(train_nf_2, Not(:precipitation_nextday))
train_nf_y = coerce!(train_nf_2, :precipitation_nextday => Multiclass).precipitation_nextday

valid_nf_x = select(train_nf_2, Not(:precipitation_nextday))[2601:end,:]
valid_nf_y = coerce!(train_nf_2, :precipitation_nextday => Multiclass).precipitation_nextday[2601:end]

n_in = length(train_nf_x[:,1])
n_out = 2


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

#saving the machine
function saving_machine(name, machine)
    filename = string("machines/", name, ".jlso")
    MLJ.save(filename, machine)
end

begin
	model_nn = NeuralNetworkClassifier(
                             builder =  MLJFlux.@builder(Chain(Dense(n_in, 64, relu),
                             Dense(64, 64),
                             Dense(64, n_out))),
                             optimiser = ADAM())
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 9),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 300, upper = 500),
                                       range(model_nn, 
                                        :(batch_size),
                                         lower = 2838, upper = n_in)])
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end

evaluate!(machine(report(mach2).best_model, train_nf_x, train_nf_y), measure = AreaUnderCurve())

saving_machine("16_nn_nf_2", mach2)
saving_predictions("16_nn_nf_2", mach2, train_nf_x, test_nf_2)
