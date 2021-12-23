### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ afdb8fbe-0f01-4691-a5b9-9a715c29d5ea
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ 504064e0-4dcb-11ec-380c-8772bf3baad2
using OpenML, MLJ, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJFlux, Flux, MLJDecisionTreeInterface, Plots, CSV, StatsPlots, LinearAlgebra, Statistics, MLJMultivariateStatsInterface, Images

# ╔═╡ e4db7640-dd92-4434-92b5-08ac005fff7e
using MLCourse

# ╔═╡ ed0af986-2afb-4121-a66a-b9f8bad76e9c
using TSne

# ╔═╡ 844fcda0-40ad-4459-937f-560f451d5557
md"# Data exploration"

# ╔═╡ f04d9f9a-4f95-4618-b6ba-0881a90b1cf5
begin
	training_data = CSV.read(joinpath(@__DIR__,"data", "trainingdata.csv"), DataFrame)
	test_data = CSV.read(joinpath(@__DIR__,"data", "testdata.csv"), DataFrame)
end;

# ╔═╡ 47dc1132-ee27-451c-a9bd-78184e100380
md"We start by changing the datatype of the precipitation_nextday column from Bool to CategoricalValue"

# ╔═╡ 57269f47-ca0d-4504-841c-2945dc13ca24
begin
	data_temp = copy(training_data)
	i = select(data_temp, :precipitation_nextday)
	coerce!(i, :precipitation_nextday => Multiclass)
	rename!(data_temp, :precipitation_nextday => "trash")

	training_data_mod = hcat(data_temp, i, makeunique=true);
	training_data_cat = select(training_data_mod, Not(:trash));
end;

# ╔═╡ 04f9e6a9-51e4-44df-a374-bfa067a03b44
md" We can handle the missing data by removing the rows including missing values"

# ╔═╡ c11f0fc4-cfee-4303-8456-634dab3c995e
begin	
	training_data_cleared = dropmissing(training_data)
	test_data_cleared = dropmissing(test_data)
	
	coerce!(training_data_cleared, :precipitation_nextday => Multiclass);
	
	training_data_cleared_x= select(training_data_cleared,Not(:precipitation_nextday))
	training_data_cleared_y = training_data_cleared.precipitation_nextday
	
end;

# ╔═╡ ed1b9c27-a1fc-4228-af18-40f2d8fcad03
length(training_data[:,1])

# ╔═╡ 535feb62-161a-47c5-a000-b5fd00fcc99b
length(training_data_cleared_x[:,1])

# ╔═╡ d1554c53-f706-4277-86fc-0f74f17ce66e
md" But as we can see, we drop half of the rows, so we should explore further how many features are missing and how we can avoid dropping half of the data"

# ╔═╡ de3797bf-443f-4427-a12a-62e7ed530c34
md" We can start by counting the maximum number of missing values in a collumn and where they are in the dataset"

# ╔═╡ 8b1b12f6-4f91-4320-ab83-0ce893ef2371
describe(training_data)

# ╔═╡ 6c860d5d-e227-4e9a-b476-0e4aaa35f014
maximum(describe(training_data).nmissing)

# ╔═╡ 8f2f04fc-b9af-41f7-86ee-7ba289976431
md" We can notice that most of the missing values are in the delta_pressure columns, and the maximum number of missing values for a column is 327, which is 10% of the number of datapoints per column. Thus we have enough information to replace the missing values by the median of the column"

# ╔═╡ 617e0349-ffe2-495f-8ac9-6f93d1190a2d
begin
	training_data_filled = MLJ.transform(fit!(machine(FillImputer(), training_data_cat)), training_data_cat)
	test_data_filled = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
	
	training_data_filled_x = select(training_data_filled, Not(:precipitation_nextday))
	training_data_filled_y = training_data_filled.precipitation_nextday
end;

# ╔═╡ 69b390e0-def4-493b-b5cd-1f3b6da8b0a3
md"We can export our new filled training set"

# ╔═╡ 1707023e-a0ec-4f16-aa03-eb2760495c9c
begin
	CSV.write("data/training_data_filled.csv", training_data_filled)
	CSV.write("data/test_data_filled.csv", test_data_filled)
end

# ╔═╡ f6b47444-0b1f-4f4d-9de7-c7f239097e4d
md"#### We can now explore or dataset, we will begin by plotting the correlation of some features to get an idea of their distribution"

# ╔═╡ 266f5238-1fa0-48a2-b85a-8b5cad2e4bdc
md"
Correlation plot code (as markdown to save loading time)
```julia
begin
	@df training_data_filled corrplot([:ABO_radiation_1 :ABO_delta_pressure_1 :ABO_air_temp_1 :ABO_sunshine_1 :ABO_wind_1 :ABO_wind_direction_1 :precipitation_nextday], grid = false, fillcolor = cgrad(), size = (700, 700))
end
```
"

# ╔═╡ eca04e01-5150-4041-8a73-569f8cb7e3fe
Images.load("pictures/corrplot_all_params.png")

# ╔═╡ 0515ad24-ed14-4139-843b-9de7d05622c7
md"

``` julia
@df training_data_filled corrplot([:ABO_radiation_1 :ABO_sunshine_1 :ABO_radiation_2 :ABO_sunshine_2 :precipitation_nextday],
                     grid = false, fillcolor = cgrad(), size = (700, 700))
```"

# ╔═╡ 5bf3e37d-7819-4187-91b2-4f33bd1c837a
md" We can see that overall most of the variables seem to be normaly distributed, and decently correlated with the response variable.
The variables Sunshine, Radiation and Air_temp are very intercorrelated, which is not surprising considering they are physically dependent on the amount of sunlight in a day"

# ╔═╡ a033f5bc-6164-486d-8d10-365671beb4a8
Images.load("pictures/corrplot_all_params_2.png")

# ╔═╡ 95124224-8a98-461d-b86b-aad2e562e553
md" The sunshine in the second measurement is more correlated to the response variable than the sunshinein the first measurement, this is coherent with what we would expect, as later measurement are closer to the measurement the next day.
We can also see that when it rains, there usually isn't a lot of sunlight"

# ╔═╡ cc6d1bfb-e142-4392-86ef-80d9ddd0e6c0
md"# Dimensionality reduction and feature extraction"

# ╔═╡ c4372b31-7e86-4bb0-abe7-c52eae8436e7
begin
	standard_mach = fit!(machine(Standardizer(), training_data_filled))
	training_data_filled_std = MLJ.transform(standard_mach, training_data_filled)
end

# ╔═╡ ed360f51-9c36-4ed9-925d-ac0d47823b8c
for col in eachcol(training_data_filled_std)
    replace!(col,NaN => 0)
end

# ╔═╡ 76ee4e3d-d8ed-4932-a601-080944290a65
begin
	PCA_mach = fit!(machine(PCA(), select(training_data_filled_std, Not(:precipitation_nextday))))
end;

# ╔═╡ 150cc1a0-2f60-4ac0-a81a-a439ad199938
report(PCA_mach).outdim

# ╔═╡ d5c0b3f1-d2ad-4131-a1a2-9806d5cbd0f0
md"
``` julia
let
    gr()
    p1 = biplot(PCA_mach)
    p2 = biplot(PCA_mach, pc = (3, 4))
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end
```"

# ╔═╡ 06407a56-c3b4-4abd-a7f1-efff6b9ba280
Images.load("pictures/pca_biplot.JPG")

# ╔═╡ 06d2d154-98b3-4603-b3fc-5583bbaeaf4b
md"The biplot is a bit too crowded to distinguish any real tendency"

# ╔═╡ e65df966-2b3c-4161-b97d-0a432798ab1c
md"
``` julia

let
    vars = report(PCA_mach).principalvars ./ report(PCA_mach).tprincipalvar

	plot(cumsum(vars),
              label = nothing, xlabel = 'predictors', ylabel = 'cumulative prop. of variance explained')

end 
```"

# ╔═╡ 2799432a-0ea1-47c3-98fd-52d94402ef51
Images.load("pictures/pca_var_explained.png")

# ╔═╡ 2c659857-3544-4c7d-8de1-bf2ea6af6aaa
md"We only need the 310 first principal components to predict 99% of the variance"

# ╔═╡ 98b04c3b-3de4-410c-a7eb-c5fa7c495139
begin
	d = MLJ.transform(PCA_mach,training_data_filled_std[:,1:end-1])
	e = DataFrame(																	precipitation = training_data_filled.precipitation_nextday)
		
	training_data_pca = hcat(e,d)
end;

# ╔═╡ c4a36662-ad9c-43c2-834b-0e5d23419c58
begin
	standard_mach_test = fit!(machine(Standardizer(), test_data_filled))
	test_data_filled_std = MLJ.transform(standard_mach, test_data_filled)
	test_data_pca = MLJ.transform(PCA_mach,test_data_filled)
end;

# ╔═╡ ff637362-88bc-4fa7-83a9-f383a441f514
md"## PCA and t-SNE for data visualization"

# ╔═╡ 206c3674-4f4b-4d8c-8933-beec8e2a60e4
m_vis_PCA = fit!(machine(PCA(maxoutdim = 2), training_data_pca));

# ╔═╡ 5042fe4c-c6b1-4124-a5e9-887eadd54c93
md"
``` julia
let
	proj = MLJ.transform(m_vis_PCA)
	scatter(proj.x1, proj.x2, color = Int.(int(training_data_filled_y)))
end
```"

# ╔═╡ 42345cbd-b901-4f07-b516-e9e4f3cd81b4
Images.load("pictures/PCA_all_params.png")

# ╔═╡ b33fcdb0-f946-4634-9648-e38152b52070
md"### Using t-SNE on filled data with every features"

# ╔═╡ 4c55b717-1563-418d-aba1-e1aa74d05090
md"
``` julia 
let
	tsne_proj = tsne(Matrix(training_data_filled_x))
	scatter(tsne_proj[:,1], tsne_proj[:,2], color = Int.(int(training_data_filled_y)))
end
```"

# ╔═╡ 39d18da7-e125-405e-b789-db6f6c024531
Images.load("pictures/tSNE_all_params.png")

# ╔═╡ 6a2ddeca-9f84-41e3-9f83-f8222e8bdd76
md"### Using t-SNE on filled data with 111 features"

# ╔═╡ bb1aa6d8-b68c-4169-b720-46793ef62e16
md"
```julia
let
	tsne_proj = tsne(Matrix(train_x))
	scatter(tsne_proj[:,1], tsne_proj[:,2], color = Int.(int(training_data_filled_y)))
end
```"

# ╔═╡ ad77a875-0766-4f2d-992e-5f6c607b6c49
Images.load("pictures/tSNE_111_features.png")

# ╔═╡ 58c7aa59-71d7-4b3e-91f8-3f46bbb39f2e
md"We can see that there is a separation of the data between the two outputs, but the border isn't clear, we might need additional feature engineering to separate the two clusters"

# ╔═╡ 8c914ea3-2ee0-48f5-8055-a750239f4cbc
md" No improvement when using a reduced amount of features"

# ╔═╡ f9dc6891-b669-471c-b102-8e50e07ab181
md"# Checking if the rows are consecutive days"

# ╔═╡ fb6e0cd9-0356-4bf4-8599-371753030e48
#Plotting the temperature for a stations during each days, row by row
begin
	temp = DataFrame(temperature = [])
	for i in 1:length(training_data.ABO_air_temp_1)
		push!(temp, [training_data_filled.ABO_air_temp_1[i]])
		push!(temp, [training_data_filled.ABO_air_temp_2[i]])
		push!(temp, [training_data_filled.ABO_air_temp_3[i]])
		push!(temp, [training_data_filled.ABO_air_temp_4[i]])
	end
end

# ╔═╡ 653e837a-5ea0-41e7-a9e9-6f7f283d6acb
begin
	scatter(temp.temperature[1:28], label = "week 1")
	scatter!(temp.temperature[29:56], label = "week 2")
end

# ╔═╡ fc5c6967-734b-4e94-80dc-9dbe5c478c19
md" There is too much variance for in temperature for the days to be consecutive, the data has been shuffled and therfor we can't use recurrent neural networks at our advantage"

# ╔═╡ fc1c3dca-d4e5-4a0f-9104-8f00b7c588e9
md"# Random forest for feature selection"

# ╔═╡ d0bddd3c-b649-452c-a6c3-597c238c6a0f
selected_features = CSV.read(joinpath(@__DIR__, "data", "selected_features.csv"), DataFrame);

# ╔═╡ 1b622113-ea3c-4760-98a5-d3db91bfd2ec
md"## Feature scores"

# ╔═╡ ed4f42af-a0bc-47be-9ea1-9fb30cad0560
features_importance = sort!(CSV.read(joinpath(@__DIR__, "data", "features_importance.csv"), DataFrame), :score, rev=true)

# ╔═╡ 1cc64552-da14-44c8-8f7c-0ac21977fb7d
md"The two most important feature types are sunshine and wind direction"

# ╔═╡ 42a218f4-d4c2-423c-91d3-7e2468be9f14
md"## Histogram comparing the score of the features on:"

# ╔═╡ 87bb379b-17af-4616-9abc-c5c0bb158e0b
md"###### - all the dataset"

# ╔═╡ 75f7e3d5-e095-45bf-a5f6-7126d03554a6
histogram(features_importance.score)

# ╔═╡ 2414664f-05b9-4831-8964-03d83e343b06
md"###### - the new dataset reduced to 111 features"

# ╔═╡ 4b912293-2028-4759-acba-1aa69868ffd7
begin
	crossed_features = semijoin(features_importance, selected_features, on = :features);
	histogram(crossed_features.score, bins = 200)
end

# ╔═╡ 1448ba93-72af-4999-b83b-08b26e247b18
md" We can see that all the features with the highest scores are kept, and most of the lowest scores are eliminated, which is in line whith what we would habe expected"

# ╔═╡ ab18ca30-5e24-486b-9a84-0e37b00b94c1
md"# Introducing new features"

# ╔═╡ 50247577-b444-4207-8cea-7b6ecc9f46fe
md"### Creating a delta-temp new feature"

# ╔═╡ 01e66da9-ab79-4b6d-923b-8f1c5f2353de
md"The intuition here is that on the days where the temperature changes the most, it is likely to be sunny outside. On the contrary, if the temperature doesn't change much during the day, it is likely to be a cloudy day"

# ╔═╡ 183ea76e-e747-4075-8229-09d191c7e61f
function f(a, b, c, d)
	return (c - a) .* abs(c - a)
end

# ╔═╡ e87a6343-5c75-4cdb-bd62-6ca438ce9ba2
#temperature increase between the first three quarters of the day
begin
	n = [string(names(training_data_filled)[i][1:3])  for i in 1:length(names(training_data_filled))-1]
	unique!(n)
	
	average_temps = copy(training_data_filled)
	test_f1 = copy(test_data_filled)
	
	for j in 1:length(n)

		transform!(average_temps, Regex(string(n[j] ,"_air_temp")) => ByRow(f))
		transform!(test_f1, Regex(string(n[j] ,"_air_temp")) => ByRow(f))
	end
	
	training_data_filled_add_feature_1 = hcat(average_temps[!,1:529], rename!(average_temps[!, 530:end], string.(n ,"_delta_temp")))
	test_1 = hcat(test_f1[!,1:528], rename!(test_f1[!, 529:end], string.(n ,"_delta_temp")))
end;

# ╔═╡ dffca61c-9eae-45a4-8754-50948979d0ac
md"
```julia
@df training_data_filled_add_feature_1 corrplot([:ZER_delta_temp :ABO_delta_temp :ALT_delta_temp :BER_delta_temp :precipitation_nextday], grid = false, fillcolor = cgrad(), size = (600, 600))
```"

# ╔═╡ c4d6535f-ba0c-4300-b5a4-0aeac7b82204
Images.load("pictures/corrplot_new_params_1.png")

# ╔═╡ 703e535a-4ec4-475a-a3fa-e84d88d99d85
md"The new feature seems to be a very strong predictor, we can try to fit and evaluate a linear regressor to evaluate the new features"

# ╔═╡ ca7d7cab-f88c-4aff-9c21-d69116d82727
mach_log_test = machine(LogisticClassifier(), training_data_filled_add_feature_1[:,530:end], training_data_filled_y) |> fit!;

# ╔═╡ 89736a82-f049-4fe3-9946-d8f70434918f
evaluate!(mach_log_test, measure = AreaUnderCurve())

# ╔═╡ da95029c-fbc9-499e-885c-30df96ce8e7e
md"We get an accuracy of 85% which is very good considering we only have 22 predictors"

# ╔═╡ 489713b7-4e31-4165-b717-c4d930bd707b
md"### Creating a delta-sunshine new feature"

# ╔═╡ e90607e5-73cb-466c-933a-9fcb9931273e
md"Here I am not sure how the trend will be, but introducing a new feature non-linearly dependent on the previous features might give us new informations"

# ╔═╡ de395826-2472-4d43-80f6-9f8f758e3541
md"Using a non-linear function:"

# ╔═╡ 0c5e5a4f-8cf8-47d0-88ca-c875728241b6
function g(a,b,c,d)
	return (c-b) .* abs(c-b)
end;

# ╔═╡ f0e607c5-1fe3-4da4-a573-68239a11f932
# difference between the sunshine at t=3 and t=2
begin
	
	sun_diff = copy(training_data_filled_add_feature_1)
	test_f2 = copy(test_1)
	for j in 1:length(n)

		transform!(sun_diff, Regex(string(n[j] ,"_sunshine")) => ByRow(g))
		transform!(test_f2, Regex(string(n[j] ,"_sunshine")) => ByRow(g))
	end
	
	training_data_filled_add_feature_2 = hcat(sun_diff[!,1:551], rename!(sun_diff[!, 552:end], string.(n ,"_sun_diff")))
	test_2 = hcat(test_f2[!,1:550], rename!(test_f2[!, 551:end], string.(n ,"_sun_diff")))
end;

# ╔═╡ 0950c80f-f8d3-4326-a3b5-249b1d0488f8
md"
```julia
@df training_data_filled_add_feature_2 corrplot([:ZER_sun_diff :ABO_sun_diff :ALT_sun_diff :BER_sun_diff :precipitation_nextday], grid = false, fillcolor = cgrad(), size = (600, 600))
```"

# ╔═╡ 404751e7-6a19-4383-81cd-42c943f7a392
Images.load("pictures/corrplot_new_params_2.png")

# ╔═╡ 7c490f95-cc21-4951-8739-5c699d23f73e
md"There seem to be some new features that are better correlated with the response variable than the others, but since we will select only the best features next, we keep the new features for the moment"

# ╔═╡ d969d586-8301-4d3f-b21e-3e03a80a7799
md"### Creating a delta-wind new feature"

# ╔═╡ 01dfb4ea-1a36-4f9d-a774-dd64ff850812
md"Creating a non-linear function"

# ╔═╡ e6330901-0fad-4de5-859f-1eaf4c0304b8
function h(a,b,c,d)
	return (max(a,b,c,d) - min(a,b,c,d))^2
end

# ╔═╡ e101a299-6c46-4721-b3d8-7b7028193d22
# variance of the wind strength during the day
begin
	sel1 = select(training_data_filled_add_feature_2, r"wind")
	sel2 = select(sel1, Not(r"direction"))

	selt1 = select(test_2, r"wind")
	selt2 = select(selt1, Not(r"direction"))
	
	wind_trans = copy(sel2)
	test_f3 = copy(selt2)

	for j in 1:length(n)

		transform!(wind_trans, Regex(string(n[j] ,"_wind")) => ByRow(h))
		transform!(test_f3, Regex(string(n[j] ,"_wind")) => ByRow(h))
	end
	
	training_data_filled_add_feature_3 = hcat(wind_trans[!,1:88], rename!(wind_trans[!, 89:end], string.(n ,"_delta_wind_st")))
	test_3 = hcat(test_f3[!,1:88], rename!(test_f3[!, 89:end], string.(n ,"_delta_wind_st")))


	training_nf = hcat(training_data_filled_add_feature_2, training_data_filled_add_feature_3[:,89:end])
	test_nf = hcat(test_2, test_3[:,89:end])
end;

# ╔═╡ 332a087f-983a-49de-90a5-8e33034a3319
md"
```julia
@df training_nf corrplot([:ZER_delta_wind_st :ABO_delta_wind_st :ALT_delta_wind_st :BER_delta_wind_st :precipitation_nextday], grid = false, fillcolor = cgrad(), size = (600, 700))
```"

# ╔═╡ 12886ef5-ae10-4086-9516-48cb4abf455a
Images.load("pictures/corrplot_new_params_3.png")

# ╔═╡ 8ff8d3be-c554-4363-bb20-a36a03b9dad3
md"The new features seem to be decently correlated with the response variable"

# ╔═╡ 6f6df539-5445-4ab3-b0d6-d97083375cc9
md"##### Exporting the new dataset containing all the new features"

# ╔═╡ 3ff1941a-a784-4e7d-94f5-95113a287afb
begin
	CSV.write("data/training_nf.csv", training_nf)
	CSV.write("data/test_nf.csv", test_nf)
end

# ╔═╡ 7e5a4c14-ab51-47e1-90b9-cea967f70c89
md"# Feature selection on new dataset including the new features"

# ╔═╡ 9812e664-cff2-4cc4-b5d9-08af2797c9fd
begin
	selected_features_nf = CSV.read(joinpath(@__DIR__, "data", "selected_features_nf.csv"), DataFrame)
	feature_importance_nf = sort!(CSV.read(joinpath(@__DIR__, "data", "feature_importance_nf.csv"), DataFrame), :score, rev=true)
	test_nf_2 = CSV.read(joinpath(@__DIR__, "data", "test_nf_2.csv"), DataFrame)
end;

# ╔═╡ 6b560332-0923-452e-b9e2-517e394cc44f
md"##### Histogram of the feature scores"

# ╔═╡ 30e9cfde-80cc-45f9-80c7-fd2f3ea4ec98
md"*For all features:*"

# ╔═╡ e02938b9-3bcc-4104-a776-3e9d7da03732
begin
	histogram(feature_importance_nf.score)
end

# ╔═╡ 23b6dcbb-f413-415a-b1a8-70cdd4a94146
md"*For the selected features*"

# ╔═╡ 7ea4ab8e-0b96-4d08-b672-3aee5f6e7fad
begin
	rename!(feature_importance_nf, :names => :features)
	crossed_features_nf = semijoin(feature_importance_nf, selected_features_nf, on = :features);
	histogram(crossed_features.score, bins = 200)
end

# ╔═╡ 20e833c9-4d6c-49e2-8f3a-e80005fe3f67
md"We can see that all the best features are kept and only a fraction of the less important features were eliminated, which is in line with what we would have expected"

# ╔═╡ e4385897-3d3b-4283-9be0-28f166b531d2
md"##### Exporting the new dataset containing only the selected features (including the new features)"

# ╔═╡ 63044562-86b6-4343-9eae-16e405d20f0e
begin
	for i in 1:length(selected_features_nf.features)
		selected_features_nf.features[i] = selected_features_nf.features[i][2:end]
	end

	train_nf_2 = training_nf[:,selected_features_nf.features]
	train_nf_2.precipitation_nextday = training_nf.precipitation_nextday

	CSV.write("data/train_nf_2", train_nf_2)
end

# ╔═╡ 96dc653a-2b91-4f39-afbc-1aa1fc40ef54
md"##### Set of new features"

# ╔═╡ 6198fada-af9f-43b1-a6b5-d4ba3132de31
selected_features_nf

# ╔═╡ 63f0ec5f-b22b-44d9-bf5a-4af64e140014
md"We can see that the new set of 390 features contains most of our artificially-created features, which is a sign of good predictibility"

# ╔═╡ efa83958-1350-40c7-b997-814f04ce5042
md"# Submissions"

# ╔═╡ 2d33243b-9c51-48b8-be14-54f4e9faa777
md"## Gradient Boosting"

# ╔═╡ 714a02a8-c900-40fd-a734-06e33768ee63
md"

#### Submission 1 with xgb on filled dataset:
best model misclassification rate = 0.159521

Score on the test data: 0.95437


best params

>num_round = 700  , 
> eta = 0.0547723  ,
>max_depth = 2  



``` julia

begin
    xgb = XGBoostClassifier()
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 5),
                            tuning = Grid(goal = 30),
                            range = [range(xgb, :eta,
                                           lower = 1e-2, upper = 0.3, scale = :log),
                                     range(xgb, :num_round, lower = 100, upper = 700),
                                     range(xgb, :max_depth, lower = 2, upper = 6)])
 
    boost_machine = machine(self_boost_mod, training_data_filled_x, training_data_filled_y) |> fit!

end;


```"

# ╔═╡ 84151290-db55-4302-b754-c59f7e389821
md"
#### Submission 2 with xgb on filled dataset

best model misclassification rate: 0.155404

Score on the test data: 0.95528

best params:

>numround = 300 , 
>eta = 0.1 , 
>maxdepth = 2 , 


definition of the model:

``` julia
begin
    xgb = XGBoostClassifier()
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 5),
                            tuning = Grid(goal = 20),
                            range = [range(xgb, :eta,
                                           lower = 1e-3, upper = 1e-1, scale = :log),
                                     range(xgb, :num_round, lower = 300, upper = 1000),
                                     range(xgb, :max_depth, lower = 1, upper = 3)])
 
    boost_machine = machine(self_boost_mod, training_data_filled_x, training_data_filled_y) |> fit!

end;
```
"

# ╔═╡ 6408759b-2a69-4dc3-9c13-8de8eb4e96a0
md"
#### Submission 3 with xgb on filled dataset

best model misclassification rate: 0.154225

Score on the test data: 0.95590

best params:

>numround = 500 , 
>eta = 0.05 , 
>maxdepth = 3 , 


definition of the model:

``` julia
# definition of the model

begin
	xgb = XGBoostClassifier(max_depth = 3)
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 7),
                            tuning = Grid(goal = 16),
                            range = [range(xgb, :eta,
                                           lower = 0.04, upper = 0.07, scale = :log),
                                   range(xgb, :num_round, lower = 400, upper = 600)])
 
    boost_machine = machine(self_boost_mod, training_data_filled_x, training_data_filled_y) |> fit!

end;
```
"

# ╔═╡ 6f7aa15a-6028-419a-b7cc-83f3528310d0
md"
#### Submission 4 with xgb on filled dataset

best model misclassification rate: 0.15894

Score on the test data: 0.95562

best params:

>numround = 440 , 
>eta = 0.0398107 , 
>maxdepth = 3 , 


definition of the model:

``` julia
# definition of the model

begin
    xgb = XGBoostClassifier()
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 5),
                            tuning = Grid(goal = 108),
                            range = [range(xgb, :eta,
                                           lower = 0.01, upper = 0.1, scale = :log),
                                   range(xgb, :num_round, lower = 200, upper = 600),
									range(xgb, :max_depth, values = [3,4,5])])
 
    boost_machine = machine(self_boost_mod, training_data_filled_x, training_data_filled_y) |> fit!

end;
```
"

# ╔═╡ be56d00c-089c-4e66-91d3-0e70b0e2acd6
md"
#### Submission 5 with xgb on filled dataset

best model misclassification rate: 0.158942

Score on the test data: 0.95431

best params:

>numround = 600 , 
>eta = 0.03 , 
>maxdepth = 3 , 


definition of the model:

``` julia
# definition of the model

begin
    xgb = XGBoostClassifier(max_depth = 3)
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 7),
                            tuning = Grid(goal = 25),
                            range = [range(xgb, :eta,
                                           lower = 0.03, upper = 0.07, scale = :log),
                                   range(xgb, :num_round, lower = 400, upper = 600)])
 
    boost_machine = machine(self_boost_mod, training_data_filled_x, training_data_filled_y) |> fit!

end;
```
"

# ╔═╡ 50bed3df-0584-4aee-8b0c-8448fffe1b1f
md"
#### Submission 6 with xgb on filled dataset

best model misclassification rate: 0.156586

Score on the test data: 0.95490

best params:

>numround = 683 , 
>eta = 0.0292402 , 
>maxdepth = 3 , 


definition of the model:

``` julia
# definition of the model

begin
    xgb = XGBoostClassifier(max_depth = 3)
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 7),
                            tuning = Grid(goal = 16),
                            range = [range(xgb, :eta,
                                           lower = 0.01, upper = 0.05, scale = :log),
                                   range(xgb, :num_round, lower = 550, upper = 750)])
 
    boost_machine = machine(self_boost_mod, training_data_filled_x, training_data_filled_y) |> fit!

end;
```
"

# ╔═╡ 617dffec-16ec-49f1-a5db-0487d78dfa6d
md"
#### Submission 7 with xgb on filled dataset

best model misclassification rate: 0.158938

Score on the test data: 0.95625

best params:

>numround = 523 , 
>eta = 0.0345455 , 
>maxdepth = 3 , 


definition of the model:

``` julia
# definition of the model

begin
    xgb = XGBoostClassifier(max_depth = 3)
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 7),
                            tuning = Grid(goal = 144),
                            range = [range(xgb, :eta, lower = 0.02, upper = 0.06),
                                   range(xgb, :num_round, lower = 450, upper = 650)])
 
    boost_machine = machine(self_boost_mod, training_data_filled_x, training_data_filled_y) |> fit!

end;
```
"

# ╔═╡ 4c39ca91-8553-4d77-9772-d9034b8efd07
md"
#### Submission 8 with xgb on filled dataset

best model misclassification rate: 0.163054

Score on the test data: 0.95603

best params:

>numround = 517 , 
>eta = 0.038 , 
>maxdepth = 3 , 


definition of the model:

``` julia
# definition of the model

begin
    xgb = XGBoostClassifier(max_depth = 3)
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 10),
                            tuning = Grid(goal = 16),
                            range = [range(xgb, :eta, lower = 0.033, upper = 0.038),
                                   range(xgb, :num_round, lower = 510, upper = 530)])
 
    boost_machine = machine(self_boost_mod, training_data_filled_x, training_data_filled_y) |> fit!

end;
```
"

# ╔═╡ 8a714a8c-7c8b-4dd8-bcc2-598f05564869
md"
#### Submission 11 with xgb on selected parameters
##### machine: ```1_xgb_all_nf```

best model AUC: 0.930219

Score on the test data: 0.95850

best params:

> eta = 0.07,
> num_rounds = 200,
> max_depth = 3


definition of the model:

``` julia
# definition of the model

begin
    xgb = XGBoostClassifier()
    mod = TunedModel(model = xgb,
                    resampling = CV(nfolds = 5),
                    tuning = Grid(goal = 64),
                    range = [range(xgb, :eta, lower = 0.01, upper = .1),
                                range(xgb, :num_round, lower = 50, upper = 500),
                                range(xgb, :max_depth, lower = 2, upper = 6)])
                                     
    m = machine(mod, train_x, train_y.precipitation_nextday)
    fit!(m)
end
```
"

# ╔═╡ 490e6e58-5286-4cf4-a758-8e592022aaed
md"
#### Submission 12 with xgb on selected parameters
##### machine: ```2_xgb_all_nf```

best model AUC: 0.931187

Score on the test data: 0.95475

best params:

> eta = 0.048,
> num_rounds = 340,
> max_depth = 3


definition of the model:

``` julia
# definition of the model

begin
    xgb = XGBoostClassifier()
    mod = TunedModel(model = xgb,
                    resampling = CV(nfolds = 5),
                    tuning = Grid(goal = 72),
                    range = [range(xgb, :eta, lower = 0.03, upper = 0.06),
                                range(xgb, :num_round, lower = 100, upper = 400),
                                range(xgb, :max_depth, values = [3,4])])
                                     
    m = machine(mod, train_x, train_y.precipitation_nextday)
    fit!(m)
end
```
"

# ╔═╡ 3784da1a-3330-4f85-9b5c-f4722da1eebb
md"
#### Submission 13 with xgb on selected parameters
##### machine: ```3_xgb_all_nf```

best model AUC: 0.931082

Score on the test data: 0.95575

best params:

> eta = 0.0497,
> num_rounds = 420,
> max_depth = 3


definition of the model:

``` julia
# definition of the model

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
```
"

# ╔═╡ 41788e47-3df4-4aae-acdd-48240ecf5a88
md"
#### Submission 15 with xgb on new features (all params included)
##### machine: ```5_xgb_all_nf```

best model AUC: 0.9351554

Score on the test data: 0.95887

best params:

> eta = 0.045,
> num_rounds = 450,
> max_depth = 3


definition of the model:

``` julia
# definition of the model

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
```
"

# ╔═╡ 07c91f52-3304-4e9b-82c8-03eabccf8d55
md"### Best gradient boosting machine"

# ╔═╡ 4a9fc78f-097e-4e7c-a84b-5bd07ba01ea0
md"
#### Submission 31 with xgb on selected features (including new features)
##### machine: ```26_xgb_nf_2```

best model AUC: 0.93791 

Score on the test data: 0.95306

params:

> max_depth = 3
> eta = 0.06,
> num_rounds = 550


definition of the model:

``` julia
# definition of the model

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
```
"

# ╔═╡ ecc6883f-a700-432e-aa6f-78dd5e172a76
md"## Random forest"

# ╔═╡ 7f83694a-b6ec-482c-87a1-74aa491d7313
md"
#### Submission 9 with Random Forest on filled dataset

best model AUC: 0.918644

Score on the test data: 0.94617

best params:

>max_depth = 7 ,
>n_subfeatures = 85 , 
>n_trees = 888 , 
>sampling_fraction = 1 , 


definition of the model:

``` julia
# definition of the model

begin
    model = RandomForestClassifier(max_depth = 7, n_subfeatures = 85)
	
    self_tuned_model = TunedModel(model = model,
                            resampling = CV(nfolds = 7),
                            tuning = Grid(goal = 81  ),
                            range = 
						 [range(model, :n_trees, lower = 100, upper = 1000),
                          range(model, :sampling_fraction, lower = 0.3, upper = 1.)])
 
    m = machine(self_tuned_model, training_data_filled_x, training_data_filled_y) |> fit!

end;
```
"

# ╔═╡ 3a860532-8d43-4d7f-b726-a375bc943580
md"## Neural networks"

# ╔═╡ c62507fd-61d9-4d71-9869-3bb79c642b9a
md"
#### Submission 20 with neural network on selected features (including new features)
##### machine: ```9_nn_nf_2```

best model AUC: 0.918 ([0.896:0.937] over 6 folds)

Score on the test data: 0.96309

params:

> two 80 neurons layers with relu
> epoch = 500,
> batch_size = 3176


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
                             builder =  MLJFlux.@builder(Chain(Dense(n_in, 80, relu),
                             Dense(80, n_out))),
                             optimiser = ADAM())
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 25),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 500, upper = 2000),
                                       range(model_nn, 
                                        :(batch_size),
                                         lower = 100, upper = n_in)])
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ 720c2f82-2d7f-4dca-af12-a09f41ae2fe2
md"
#### Submission 21 with neural network on selected features (including new features)
##### machine: ```10_nn_nf_2```

best model AUC: 0.917 ([0.897:0.928] over 6 folds)

validation AUC: 0.909 ([0.868, 0.959] over 6 folds)

Score on the test data: 0.95750

params:

> two 40 neurons layers with relu
> epoch = 200,
> batch_size = 2600


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
                             builder =  MLJFlux.@builder(Chain(Dense(n_in, 40, relu),
                             Dense(40, n_out))),
                             optimiser = ADAM())
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 9),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 200, upper = 900),
                                       range(model_nn, 
                                        :(batch_size),
                                         lower = 2000, upper = n_in)])
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ 051adb23-2a96-409d-96c3-9cc60d377dba
md"
#### Submission 23 with neural network on selected features (including new features)
##### machine: ```12_nn_nf_2```

best model AUC: 0.921 ( [0.91, 0.926] over 6 folds)

Score on the test data: 0.96065

params:

> two 100 neurons layers with relu
> epoch = 200,
> batch_size = 3176


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
                             builder =  MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
                             Dense(100, n_out))),
                             optimiser = ADAM())
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 9),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 200, upper = 900),
                                       range(model_nn, 
                                        :(batch_size),
                                         lower = 2000, upper = n_in)])
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ 7f98dc7e-c934-4eea-9ba3-b9d8d9cdda02
md"
#### Submission 24 with neural network on selected features (including new features)
##### machine: ```13_nn_nf_2```

best model AUC: 0.914 ( [ 0.895, 0.927] over 6 folds)

Score on the test data: 0.96753

params:

> three 60 neurons layers with relu
> epoch = 200,
> batch_size = 2838


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
                             builder =  MLJFlux.@builder(Chain(Dense(n_in, 60, relu),
                             Dense(60, 60),
                             Dense(60, n_out))),
                             optimiser = ADAM())
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 9),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 200, upper = 1500),
                                       range(model_nn, 
                                        :(batch_size),
                                         lower = 2500, upper = n_in)])
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ 2b492abe-848a-4ff7-9cfe-d372a40dbace
md"
#### Submission 25 with neural network on selected features (including new features)
##### machine: ```16_nn_nf_2```

best model AUC: 0.916 (  [0.901, 0.928] over 6 folds)

Score on the test data: 0.95221

params:

> three 64 neurons layers with relu
> epoch = 300,
> batch_size = 3007


definition of the model:

``` julia
# definition of the model

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
```
"

# ╔═╡ f29e5937-6b42-496c-a963-02b52604f911
md"
#### Submission 26 with neural network on selected features (including new features)
##### machine: ```20_nn_nf_2```

best model AUC: 0.925 ( [0.919, 0.935] over 6 folds)

Score on the test data: 0.95768

params:

> 128 -> 64 neuron layers with relu
> epoch = 56,
> batch_size = 3176


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
                             builder =  MLJFlux.@builder(Chain(Dense(n_in, 128, relu),
                             Dense(128, 64, relu),
							 Dense(64, n_out))),
                             batch_size = n_in,
                             optimiser = ADAM()) 
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 10),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 1, upper = 500)],
                              measure = auc)
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ b4295765-d7a2-4fdb-83cd-a8b6dfcdfa7d


# ╔═╡ b1704e94-13b3-4ddf-84f9-3e1a038ed44d


# ╔═╡ fae7f18b-332d-4d7e-9ab8-deb84ab5cf8f


# ╔═╡ 3a8499b6-0826-4253-bcfc-6148c0ae0197
md"### Best neural network"

# ╔═╡ 1279b0ac-e4b8-49b5-b8a5-33513c7f0b3e
md"
#### Submission 27 with neural network on selected features (including new features)
##### machine: ```21_nn_nf_2```
best model AUC: 0.928 ( [0.917, 0.943] over 6 folds)

Score on the test data: 0.95856

params:

> 256 -> 128 neuron layers with relu
> epoch = 30,
> batch_size = 390


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
                             builder =  MLJFlux.@builder(Chain(Dense(n_in, 128, relu),
                             Dense(128, 64, relu),
							 Dense(64, n_out))),
                             batch_size = n_in,
                             optimiser = ADAM()) 
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 10),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 1, upper = 500)],
                              measure = auc)
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ e7f3cfc9-47c4-4265-af13-b97cbd7f3aed


# ╔═╡ be47a66e-c353-44fb-94a9-5f5b1a42a873


# ╔═╡ 711c5746-6bd8-41c8-95ae-73107f5d68a6


# ╔═╡ 2a31b142-3f68-41b4-83e3-eea8c59a7895
md"
#### Submission 28 with neural network on selected features (including new features)
##### machine: ```22_nn_nf_2```

best model AUC: 0.924 ( [0.913, 0.931] over 6 folds)

Score on the test data: 0.96550

params:

> 256 -> 128 neuron layers with relu
> epoch = 122,
> batch_size = 390


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
                             builder =  MLJFlux.@builder(Chain(Dense(n_in, 256, relu),
                             Dense(256, 128, relu),
							 Dense(128, n_out))),
                             batch_size = n_in,
                             optimiser = ADAM()) 
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 10),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 100, upper = 200)],
                              measure = auc)
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ 0213d9de-44ac-46eb-8f02-2e9ca4beac58
md"
#### Submission 32 with neural network on selected features (including new features)
##### machine: ```27_nn_nf_2```

best model AUC: 0.921 ( [0.9 : 0.934] over 6 folds)

Score on the test data: 0.95956

params:

> 64 -> 64 neuron layers with relu
> epoch = 50,
> batch_size = 390
> alpha = 1
> lambda = 0


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
							builder =  MLJFlux.@builder(Chain(Dense(n_in, 64, relu),
                             Dense(64, 64, relu),
							 Dense(64, n_out))),
                             batch_size = n_in,
                             optimiser = ADAM()) 
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 64),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 50, upper = 500),
									   range(model_nn,
									     :(alpha),
									     lower = 0, upper = 1),
									   range(model_nn,
									     :(lambda),
									     lower = 0, upper = 100)],
                              measure = auc)
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ c9807dd2-f547-4750-bc39-9325f8862092
md"
#### Submission 36 with neural network on selected features (including new features)
##### machine: ```30_nn_nf_2```

best model AUC: 0.923 ([0.915, 0.932] over 6 folds)

Score on the test data: 0.96396

params:

> 64 hidden neuron with relu
> epoch = 17,
> batch_size = 390
> alpha = 1
> lambda = 0


definition of the model:

``` julia
# definition of the model

begin
	model_nn = NeuralNetworkClassifier(
							builder =  MLJFlux.@builder(Chain(Dense(n_in, 64, relu),
							 Dense(64, n_out))),
                             batch_size = n_in,
							 alpha = 1.,
                             optimiser = ADAM()) 
	tuned_model = TunedModel(model = model_nn,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 105),
	                          range = [range(model_nn,
									     :(epochs),
									     lower = 1, upper = 70),
									   range(model_nn,
									     :(lambda),
									     values = [0.,0.01,0.1])],
                              measure = auc)
	mach2 = fit!(machine(tuned_model,train_nf_x, train_nf_y))
end
```
"

# ╔═╡ 59c7a5d3-30e0-45e6-857b-c8ba4d4cb3c4
md"
#### Submission 40 with neural network on selected features (including new features)
##### machine: ```40_nn_lin_nf_2```

best model AUC: 0.918 ([0.915, 0.932] over 6 folds)

Score on the test data: 0.94259

params:

> no hidden layer
> epoch = 54,
> batch_size = 390



definition of the model:

``` julia
# definition of the model

begin
	model_nn_2 = NeuralNetworkClassifier(
							builder =  MLJFlux.@builder(Chain(Dense(n_in, n_out))),
                            batch_size = n_in,
                            optimiser = ADAM()) 

	tuned_model_2 = TunedModel(model = model_nn_2,
							  resampling = CV(nfolds = 5),
                              tuning = Grid(goal = 50),
	                          range = [range(model_nn_2,
									     :(epochs),
									     lower = 50, upper = 250)],
                              measure = auc)

	mach3 = fit!(machine(tuned_model_2, train_nf_2_x, train_nf_2_y))
end
```
"

# ╔═╡ 803f40f0-d67b-40a9-bd4d-41fe1eacd0e9
md"# Loading and evaluating models"

# ╔═╡ e8de62e4-e416-4c6f-94a7-dc503cebb813
md" Here is a space to load, control and assess the validity of models trained in the scripts. The machine, the prediction on training and the prediction on the test (submission) can be loaded in the cell *just below*"

# ╔═╡ 25123914-bef5-4023-8e1c-e77e555410de
begin
	train_x = CSV.read(joinpath("data/train_x.csv"), DataFrame)
	train_y = CSV.read(joinpath("data/train_y.csv"), DataFrame)
	test = CSV.read(joinpath("data/test.csv"), DataFrame);
end;

# ╔═╡ 220edda4-f439-49cb-b7a8-f8b3853d8377
function fetch(n)
	return machine(string("machines/", n ,".jlso")), CSV.read(joinpath(string("predictions/prediction_train_", n, ".csv")), DataFrame), CSV.read(joinpath(string("submissions/submission_",n,".csv")), DataFrame)
end;

# ╔═╡ d6541a57-5cf3-4e32-93eb-d3ea7383cf7a
md"To load those files, you only need to put in string form the name of the machine as a parameter of the *fetch* function. This function can sometime bug, just running the cell again should correct it"

# ╔═╡ de56130e-e397-4412-b175-5ddce3f0c406
m, prediction_train, submission = fetch("21_nn_nf_2");

# ╔═╡ 1e2306c7-068e-4b91-bb0a-673ead8b96a0
md"##### Parameter comparasion (only works with tunned models)"

# ╔═╡ 55c4c4fe-fa9b-476b-a744-1ed319f3e937
plot(m)

# ╔═╡ 42231026-18f8-4367-bf27-ee42c042450a
md"#### Optimal parameters found during the training"

# ╔═╡ 022ebee0-8d70-47a0-ade5-b50bfcc2b849
fitted_params(m)

# ╔═╡ d75c164e-cef7-42e0-b846-96a689de45a8
md"#### Prediction on training set
This allows us to detect if a model overfits the training set. The blue points represent the prediction and the red points the true labels"

# ╔═╡ cec1f5ee-5a3c-4b2b-b605-b7f4b658e7ef
begin
	scatter(prediction_train.p[1:end])
	scatter!(convert(AbstractArray{Float32}, training_data_filled_y)[1:end])

end

# ╔═╡ 40256e2d-93b5-4899-a823-2cf39eeb401b
md"#### Prediction on the test set
This allows us to detect outliers and assess the overall quality of our prediction, even if we don't have the true labels and therefor can't control the precision quantitatively"

# ╔═╡ 4db6ac3a-d187-4a45-aee3-bf3347fcbda7
submission

# ╔═╡ 22f22845-6033-446c-838c-0ae4410fc2cc
begin
	limits = 0
	for i in 1:1200
		if (submission.precipitation_nextday[i] > 0.4) && (submission.precipitation_nextday[i] < 0.6)
			limits += 1
		end
	end
end

# ╔═╡ 12da20d1-be32-40fb-84eb-782337c84e75
md"##### Number of predictions in the *grey zone* between 0.4 and 0.6"

# ╔═╡ 84d9bb72-3337-4141-9b03-e8c1d5e663fd
limits

# ╔═╡ f793efdb-f5d0-4611-91af-3690afb0c683
md"##### Prediction on the test set"

# ╔═╡ f83255f8-7a92-4ec6-bfb4-835b0532b9f0
begin
	scatter(submission.precipitation_nextday[1:end])
end

# ╔═╡ Cell order:
# ╠═504064e0-4dcb-11ec-380c-8772bf3baad2
# ╟─844fcda0-40ad-4459-937f-560f451d5557
# ╠═f04d9f9a-4f95-4618-b6ba-0881a90b1cf5
# ╟─47dc1132-ee27-451c-a9bd-78184e100380
# ╠═57269f47-ca0d-4504-841c-2945dc13ca24
# ╟─04f9e6a9-51e4-44df-a374-bfa067a03b44
# ╠═c11f0fc4-cfee-4303-8456-634dab3c995e
# ╠═ed1b9c27-a1fc-4228-af18-40f2d8fcad03
# ╠═535feb62-161a-47c5-a000-b5fd00fcc99b
# ╟─d1554c53-f706-4277-86fc-0f74f17ce66e
# ╟─de3797bf-443f-4427-a12a-62e7ed530c34
# ╟─8b1b12f6-4f91-4320-ab83-0ce893ef2371
# ╠═6c860d5d-e227-4e9a-b476-0e4aaa35f014
# ╟─8f2f04fc-b9af-41f7-86ee-7ba289976431
# ╠═617e0349-ffe2-495f-8ac9-6f93d1190a2d
# ╟─69b390e0-def4-493b-b5cd-1f3b6da8b0a3
# ╠═1707023e-a0ec-4f16-aa03-eb2760495c9c
# ╟─f6b47444-0b1f-4f4d-9de7-c7f239097e4d
# ╟─266f5238-1fa0-48a2-b85a-8b5cad2e4bdc
# ╟─eca04e01-5150-4041-8a73-569f8cb7e3fe
# ╟─0515ad24-ed14-4139-843b-9de7d05622c7
# ╟─5bf3e37d-7819-4187-91b2-4f33bd1c837a
# ╠═a033f5bc-6164-486d-8d10-365671beb4a8
# ╟─95124224-8a98-461d-b86b-aad2e562e553
# ╟─cc6d1bfb-e142-4392-86ef-80d9ddd0e6c0
# ╠═c4372b31-7e86-4bb0-abe7-c52eae8436e7
# ╠═ed360f51-9c36-4ed9-925d-ac0d47823b8c
# ╠═76ee4e3d-d8ed-4932-a601-080944290a65
# ╠═150cc1a0-2f60-4ac0-a81a-a439ad199938
# ╠═e4db7640-dd92-4434-92b5-08ac005fff7e
# ╟─d5c0b3f1-d2ad-4131-a1a2-9806d5cbd0f0
# ╟─06407a56-c3b4-4abd-a7f1-efff6b9ba280
# ╟─06d2d154-98b3-4603-b3fc-5583bbaeaf4b
# ╟─e65df966-2b3c-4161-b97d-0a432798ab1c
# ╟─2799432a-0ea1-47c3-98fd-52d94402ef51
# ╟─2c659857-3544-4c7d-8de1-bf2ea6af6aaa
# ╠═98b04c3b-3de4-410c-a7eb-c5fa7c495139
# ╠═c4a36662-ad9c-43c2-834b-0e5d23419c58
# ╟─ff637362-88bc-4fa7-83a9-f383a441f514
# ╟─206c3674-4f4b-4d8c-8933-beec8e2a60e4
# ╟─5042fe4c-c6b1-4124-a5e9-887eadd54c93
# ╟─42345cbd-b901-4f07-b516-e9e4f3cd81b4
# ╠═ed0af986-2afb-4121-a66a-b9f8bad76e9c
# ╟─b33fcdb0-f946-4634-9648-e38152b52070
# ╟─4c55b717-1563-418d-aba1-e1aa74d05090
# ╟─39d18da7-e125-405e-b789-db6f6c024531
# ╟─6a2ddeca-9f84-41e3-9f83-f8222e8bdd76
# ╟─bb1aa6d8-b68c-4169-b720-46793ef62e16
# ╟─ad77a875-0766-4f2d-992e-5f6c607b6c49
# ╟─58c7aa59-71d7-4b3e-91f8-3f46bbb39f2e
# ╟─8c914ea3-2ee0-48f5-8055-a750239f4cbc
# ╟─f9dc6891-b669-471c-b102-8e50e07ab181
# ╠═fb6e0cd9-0356-4bf4-8599-371753030e48
# ╟─653e837a-5ea0-41e7-a9e9-6f7f283d6acb
# ╟─fc5c6967-734b-4e94-80dc-9dbe5c478c19
# ╟─fc1c3dca-d4e5-4a0f-9104-8f00b7c588e9
# ╠═d0bddd3c-b649-452c-a6c3-597c238c6a0f
# ╟─1b622113-ea3c-4760-98a5-d3db91bfd2ec
# ╟─ed4f42af-a0bc-47be-9ea1-9fb30cad0560
# ╟─1cc64552-da14-44c8-8f7c-0ac21977fb7d
# ╟─42a218f4-d4c2-423c-91d3-7e2468be9f14
# ╟─87bb379b-17af-4616-9abc-c5c0bb158e0b
# ╟─75f7e3d5-e095-45bf-a5f6-7126d03554a6
# ╟─2414664f-05b9-4831-8964-03d83e343b06
# ╟─4b912293-2028-4759-acba-1aa69868ffd7
# ╟─1448ba93-72af-4999-b83b-08b26e247b18
# ╟─ab18ca30-5e24-486b-9a84-0e37b00b94c1
# ╟─50247577-b444-4207-8cea-7b6ecc9f46fe
# ╟─01e66da9-ab79-4b6d-923b-8f1c5f2353de
# ╠═183ea76e-e747-4075-8229-09d191c7e61f
# ╠═e87a6343-5c75-4cdb-bd62-6ca438ce9ba2
# ╟─dffca61c-9eae-45a4-8754-50948979d0ac
# ╟─c4d6535f-ba0c-4300-b5a4-0aeac7b82204
# ╟─703e535a-4ec4-475a-a3fa-e84d88d99d85
# ╠═ca7d7cab-f88c-4aff-9c21-d69116d82727
# ╠═89736a82-f049-4fe3-9946-d8f70434918f
# ╟─da95029c-fbc9-499e-885c-30df96ce8e7e
# ╟─489713b7-4e31-4165-b717-c4d930bd707b
# ╟─e90607e5-73cb-466c-933a-9fcb9931273e
# ╟─de395826-2472-4d43-80f6-9f8f758e3541
# ╠═0c5e5a4f-8cf8-47d0-88ca-c875728241b6
# ╠═f0e607c5-1fe3-4da4-a573-68239a11f932
# ╟─0950c80f-f8d3-4326-a3b5-249b1d0488f8
# ╟─404751e7-6a19-4383-81cd-42c943f7a392
# ╟─7c490f95-cc21-4951-8739-5c699d23f73e
# ╟─d969d586-8301-4d3f-b21e-3e03a80a7799
# ╟─01dfb4ea-1a36-4f9d-a774-dd64ff850812
# ╠═e6330901-0fad-4de5-859f-1eaf4c0304b8
# ╠═e101a299-6c46-4721-b3d8-7b7028193d22
# ╟─332a087f-983a-49de-90a5-8e33034a3319
# ╟─12886ef5-ae10-4086-9516-48cb4abf455a
# ╟─8ff8d3be-c554-4363-bb20-a36a03b9dad3
# ╟─6f6df539-5445-4ab3-b0d6-d97083375cc9
# ╠═3ff1941a-a784-4e7d-94f5-95113a287afb
# ╟─7e5a4c14-ab51-47e1-90b9-cea967f70c89
# ╠═9812e664-cff2-4cc4-b5d9-08af2797c9fd
# ╟─6b560332-0923-452e-b9e2-517e394cc44f
# ╟─30e9cfde-80cc-45f9-80c7-fd2f3ea4ec98
# ╟─e02938b9-3bcc-4104-a776-3e9d7da03732
# ╟─23b6dcbb-f413-415a-b1a8-70cdd4a94146
# ╟─7ea4ab8e-0b96-4d08-b672-3aee5f6e7fad
# ╟─20e833c9-4d6c-49e2-8f3a-e80005fe3f67
# ╟─e4385897-3d3b-4283-9be0-28f166b531d2
# ╠═63044562-86b6-4343-9eae-16e405d20f0e
# ╟─96dc653a-2b91-4f39-afbc-1aa1fc40ef54
# ╟─6198fada-af9f-43b1-a6b5-d4ba3132de31
# ╟─63f0ec5f-b22b-44d9-bf5a-4af64e140014
# ╟─efa83958-1350-40c7-b997-814f04ce5042
# ╟─2d33243b-9c51-48b8-be14-54f4e9faa777
# ╟─714a02a8-c900-40fd-a734-06e33768ee63
# ╟─84151290-db55-4302-b754-c59f7e389821
# ╟─6408759b-2a69-4dc3-9c13-8de8eb4e96a0
# ╟─6f7aa15a-6028-419a-b7cc-83f3528310d0
# ╟─be56d00c-089c-4e66-91d3-0e70b0e2acd6
# ╟─50bed3df-0584-4aee-8b0c-8448fffe1b1f
# ╟─617dffec-16ec-49f1-a5db-0487d78dfa6d
# ╟─4c39ca91-8553-4d77-9772-d9034b8efd07
# ╟─8a714a8c-7c8b-4dd8-bcc2-598f05564869
# ╟─490e6e58-5286-4cf4-a758-8e592022aaed
# ╟─3784da1a-3330-4f85-9b5c-f4722da1eebb
# ╟─41788e47-3df4-4aae-acdd-48240ecf5a88
# ╟─07c91f52-3304-4e9b-82c8-03eabccf8d55
# ╟─4a9fc78f-097e-4e7c-a84b-5bd07ba01ea0
# ╟─ecc6883f-a700-432e-aa6f-78dd5e172a76
# ╟─7f83694a-b6ec-482c-87a1-74aa491d7313
# ╟─3a860532-8d43-4d7f-b726-a375bc943580
# ╟─c62507fd-61d9-4d71-9869-3bb79c642b9a
# ╟─720c2f82-2d7f-4dca-af12-a09f41ae2fe2
# ╟─051adb23-2a96-409d-96c3-9cc60d377dba
# ╟─7f98dc7e-c934-4eea-9ba3-b9d8d9cdda02
# ╟─2b492abe-848a-4ff7-9cfe-d372a40dbace
# ╟─f29e5937-6b42-496c-a963-02b52604f911
# ╟─b4295765-d7a2-4fdb-83cd-a8b6dfcdfa7d
# ╟─b1704e94-13b3-4ddf-84f9-3e1a038ed44d
# ╟─fae7f18b-332d-4d7e-9ab8-deb84ab5cf8f
# ╟─3a8499b6-0826-4253-bcfc-6148c0ae0197
# ╟─1279b0ac-e4b8-49b5-b8a5-33513c7f0b3e
# ╟─e7f3cfc9-47c4-4265-af13-b97cbd7f3aed
# ╟─be47a66e-c353-44fb-94a9-5f5b1a42a873
# ╟─711c5746-6bd8-41c8-95ae-73107f5d68a6
# ╟─2a31b142-3f68-41b4-83e3-eea8c59a7895
# ╟─0213d9de-44ac-46eb-8f02-2e9ca4beac58
# ╟─c9807dd2-f547-4750-bc39-9325f8862092
# ╟─59c7a5d3-30e0-45e6-857b-c8ba4d4cb3c4
# ╟─803f40f0-d67b-40a9-bd4d-41fe1eacd0e9
# ╟─e8de62e4-e416-4c6f-94a7-dc503cebb813
# ╟─25123914-bef5-4023-8e1c-e77e555410de
# ╟─220edda4-f439-49cb-b7a8-f8b3853d8377
# ╟─d6541a57-5cf3-4e32-93eb-d3ea7383cf7a
# ╠═de56130e-e397-4412-b175-5ddce3f0c406
# ╟─1e2306c7-068e-4b91-bb0a-673ead8b96a0
# ╟─55c4c4fe-fa9b-476b-a744-1ed319f3e937
# ╟─42231026-18f8-4367-bf27-ee42c042450a
# ╟─022ebee0-8d70-47a0-ade5-b50bfcc2b849
# ╟─d75c164e-cef7-42e0-b846-96a689de45a8
# ╟─cec1f5ee-5a3c-4b2b-b605-b7f4b658e7ef
# ╟─40256e2d-93b5-4899-a823-2cf39eeb401b
# ╟─4db6ac3a-d187-4a45-aee3-bf3347fcbda7
# ╟─22f22845-6033-446c-838c-0ae4410fc2cc
# ╟─12da20d1-be32-40fb-84eb-782337c84e75
# ╠═84d9bb72-3337-4141-9b03-e8c1d5e663fd
# ╟─f793efdb-f5d0-4611-91af-3690afb0c683
# ╟─f83255f8-7a92-4ec6-bfb4-835b0532b9f0
# ╟─afdb8fbe-0f01-4691-a5b9-9a715c29d5ea
