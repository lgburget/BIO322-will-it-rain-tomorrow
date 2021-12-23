using Pkg
Pkg.activate(".")

using DataFrames, Plots, CSV, Random

#saving the machine
function saving_machine(name, machine)
    filename = string("machines/", name, ".jlso")
    MLJ.save(filename, machine)
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
 