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
using OpenML, MLJ, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJFlux, Flux, MLJDecisionTreeInterface, Plots, CSV, StatsPlots, LinearAlgebra, Statistics, MLJMultivariateStatsInterface

# ╔═╡ d6f8805a-8cbf-4caf-beb0-5622e6eaaf62
using Images

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

# ╔═╡ 57269f47-ca0d-4504-841c-2945dc13ca24
begin
	data_temp = copy(training_data)
	i = select(data_temp, :precipitation_nextday)
	coerce!(i, :precipitation_nextday => Multiclass)
	rename!(data_temp, :precipitation_nextday => "trash")

	training_data_mod = hcat(data_temp, i, makeunique=true);
	training_data_cat = select(training_data_mod, Not(:trash));
end;

# ╔═╡ c11f0fc4-cfee-4303-8456-634dab3c995e
begin	
	training_data_cleared = dropmissing(training_data)
	test_data_cleared = dropmissing(test_data)
	
	coerce!(training_data_cleared, :precipitation_nextday => Multiclass);
	
	training_data_cleared_x= select(training_data_cleared,Not(:precipitation_nextday))
	training_data_cleared_y = training_data_cleared.precipitation_nextday
	
end;

# ╔═╡ 6c860d5d-e227-4e9a-b476-0e4aaa35f014


# ╔═╡ 617e0349-ffe2-495f-8ac9-6f93d1190a2d
begin
	training_data_filled = MLJ.transform(fit!(machine(FillImputer(), training_data_cat)), training_data_cat)
	test_data_filled = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
	
	training_data_filled_x = select(training_data_filled, Not(:precipitation_nextday))
	training_data_filled_y = training_data_filled.precipitation_nextday
end;

# ╔═╡ 1707023e-a0ec-4f16-aa03-eb2760495c9c
begin
	CSV.write("data/training_data_filled.csv", training_data_filled)
	CSV.write("data/test_data_filled.csv", test_data_filled)
end

# ╔═╡ 4b148eef-5521-47c0-8ce0-6354790a24b2
size(training_data_cleared)

# ╔═╡ 591f7943-1b0d-4cec-9cc9-5d846735884f
size(training_data_filled)

# ╔═╡ 266f5238-1fa0-48a2-b85a-8b5cad2e4bdc
md"
Correlation plot code (as markdown to save computation time)
```julia
begin
	@df training_data_filled corrplot([:ABO_radiation_1 :ABO_delta_pressure_1 :ABO_air_temp_1 :ABO_sunshine_1 :ABO_wind_1 :ABO_wind_direction_1 :precipitation_nextday], grid = false, fillcolor = cgrad(), size = (700, 700))
end
```
"

# ╔═╡ 1851b953-62d2-4dcd-b19b-9b3ecf7dd667
maximum(describe(training_data).nmissing)

# ╔═╡ 0515ad24-ed14-4139-843b-9de7d05622c7
md"

``` julia
@df training_data_filled corrplot([:ABO_radiation_1 :ABO_sunshine_1 :ABO_radiation_2 :ABO_sunshine_2 :precipitation_nextday],
                     grid = false, fillcolor = cgrad(), size = (700, 700))
```"

# ╔═╡ a033f5bc-6164-486d-8d10-365671beb4a8
Images.load("pictures/corrplot_all_params.png")

# ╔═╡ cc6d1bfb-e142-4392-86ef-80d9ddd0e6c0
md"## PCA to reduce the noise"

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
md"# PCA and t-SNE for data visualization"

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

# ╔═╡ b33fcdb0-f946-4634-9648-e38152b52070
md"### using t-SNE on filled data with every features"

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
md"### using t-SNE on filled data with 111 features"

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

# ╔═╡ ddbfb8aa-8628-4b84-9ba9-06adc09a2b1d
training_data

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
md" Too much variance, the data has been shuffled"

# ╔═╡ a42a4137-ff6e-4a61-9a7a-d8dd5a4bc8c6
md"# Describing Datasets"

# ╔═╡ 3e4bc22f-f005-4257-90ae-c6fc6cf6f02b
describe(training_data_cleared)

# ╔═╡ c2b766fc-3f93-4b99-96fc-3eb918170463
describe(training_data_filled)

# ╔═╡ 6fd4c29e-1400-47cd-b8e5-82d2520b47df
describe(training_data_pca)

# ╔═╡ fc1c3dca-d4e5-4a0f-9104-8f00b7c588e9
md"# Random forest for feature selection"

# ╔═╡ d0bddd3c-b649-452c-a6c3-597c238c6a0f
selected_features = CSV.read(joinpath(@__DIR__, "data", "selected_features.csv"), DataFrame);

# ╔═╡ ed4f42af-a0bc-47be-9ea1-9fb30cad0560
features_importance = sort!(CSV.read(joinpath(@__DIR__, "data", "features_importance.csv"), DataFrame), :score, rev=true)

# ╔═╡ 1cc64552-da14-44c8-8f7c-0ac21977fb7d
md"The two most important data types are sunshine and wind direction"

# ╔═╡ 49217757-5c91-471b-91d9-63f68793807b
training_data_filled.ZER_sunshine_3

# ╔═╡ 42db0c60-70d7-472b-b603-3a89b45cdc50
begin
	a = select(training_data_filled, :precipitation_nextday, r"ZER")
	b = select(a, :precipitation_nextday, r"3")
end

# ╔═╡ 246e86fe-b1e9-41ad-8628-700385fd21f7
begin
	scatter((training_data_filled.ZER_sunshine_3[1:100]), training_data_filled.precipitation_nextday[1:100])
end

# ╔═╡ 3baf6d35-7277-4457-8faf-ecea6da3639b
begin
	scatter(training_data_filled.ZER_sunshine_3[1:100].^3, training_data_filled.precipitation_nextday[1:100])
end

# ╔═╡ 3856d15e-b094-43a5-9061-d1790b73a48f
b_std = MLJ.transform(fit!(machine(Standardizer(), b)), b)

# ╔═╡ f7ff1079-2505-44f9-835e-49cb217ac02e
begin
	scatter(b_std.ZER_radiation_3[1:1000].^3 , b_std.precipitation_nextday[1:1000])
end

# ╔═╡ 75f7e3d5-e095-45bf-a5f6-7126d03554a6
histogram(features_importance.score)

# ╔═╡ 424a81a4-2580-481b-81a4-8ffa7ce635f9
crossed_features = semijoin(features_importance, selected_features, on = :features);

# ╔═╡ 4b912293-2028-4759-acba-1aa69868ffd7
histogram(crossed_features.score, bins = 200)

# ╔═╡ ab18ca30-5e24-486b-9a84-0e37b00b94c1
md"# Introducing new features"

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
md"Seems to be a decent predictor"

# ╔═╡ 9a2f0053-34af-4ae1-b0f6-cdf6b6f69ce1
training_nf[:,530]

# ╔═╡ 3ff1941a-a784-4e7d-94f5-95113a287afb
#exporting the new dataset containing all the new features
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
end

# ╔═╡ e02938b9-3bcc-4104-a776-3e9d7da03732
begin
	histogram(feature_importance_nf.score)
end

# ╔═╡ 7ea4ab8e-0b96-4d08-b672-3aee5f6e7fad
begin
	rename!(feature_importance_nf, :names => :features)
	crossed_features_nf = semijoin(feature_importance_nf, selected_features_nf, on = :features);
	histogram(crossed_features.score, bins = 200)
end

# ╔═╡ 63044562-86b6-4343-9eae-16e405d20f0e
begin
	for i in 1:length(selected_features_nf.features)
		selected_features_nf.features[i] = selected_features_nf.features[i][2:end]
	end

	train_nf_2 = training_nf[:,selected_features_nf.features]
	train_nf_2.precipitation_nextday = training_nf.precipitation_nextday

	CSV.write("data/train_nf_2", train_nf_2)
end

# ╔═╡ 891b3a58-74eb-40c6-b25f-7203f354bd63


# ╔═╡ c79899c5-c88b-4d95-82fb-7578201f7eff
m3 = machine(LogisticClassifier(), select(train_nf_2, Not(:precipitation_nextday))[1:end,:], train_nf_2.precipitation_nextday[1:end]) |> fit!

# ╔═╡ 10c06962-5d01-4094-a182-b45fbdb65ab7
evaluate!(m3, resampling = CV(nfolds = 3), measure = AreaUnderCurve())

# ╔═╡ f892effe-2e7f-484e-b212-43a22c66d2bd
p1 = predict(m3, train_nf_2)

# ╔═╡ 4f20e66a-d630-4ffa-873a-0329f062244f
begin
	plot(broadcast(pdf, p1, true)[1:200])
	plot!(convert(AbstractArray{Float32}, training_data_filled_y)[1:200])
end

# ╔═╡ 980c8c10-5cf3-48d7-a93a-4e663f030cec
begin
	out = (id = 1:1200, precipitation_nextday = broadcast(pdf, predict(m3, test_nf_2), true))
	CSV.write("submissions/submission_8_logistic_nf_2.csv", out)
end

# ╔═╡ 6198fada-af9f-43b1-a6b5-d4ba3132de31
selected_features_nf

# ╔═╡ efa83958-1350-40c7-b997-814f04ce5042
md"# Submissions"

# ╔═╡ 8332735d-30b2-4ba9-a984-e2792d819adf
md"
```julia
begin
	xgb = XGBoostClassifier()
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 5),
                            tuning = Grid(goal = 25),
                            range = [range(xgb, :eta,
                                           lower = 1e-2, upper = .1, scale = :log),
                                     range(xgb, :num_round, lower = 50, upper = 500),
                                     range(xgb, :max_depth, lower = 2, upper = 6)])
	
	boost_machine = machine(self_boost_mod, training_data_cleared_x, training_data_cleared_y) |> fit!

end;
```"

# ╔═╡ 714a02a8-c900-40fd-a734-06e33768ee63
md"

### First submission:
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
### Second submission

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
### Third submission

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
### Big parameter comparaison

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
### Fith Submission

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
### Sixth Submission

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
### Seventh Submission

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
### Eighth Submission

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

# ╔═╡ 79a1647e-e34f-4e32-bed7-3c2864717fdb
md"
### 9th Submission

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

# ╔═╡ 7235988b-2a04-4b4a-ae05-79d3b8faf284
md"
### First PCA parameters test

best model AUC: 0.

Score on the test data: 0.

best params:

> eta = ,
> num_rounds = ,
> max_depth


definition of the model:

``` julia
# definition of the model

begin
	training_data_pca_x = select(training_data_pca, Not(:precipitation))
	training_data_pca_y = training_data_pca.precipitation
	

    xgb = XGBoostClassifier()
    self_boost_mod = TunedModel(model = xgb,
                            resampling = CV(nfolds = 7),
                            tuning = Grid(goal = 75),
                            range = [range(xgb, :eta, lower = 0.01, upper = 0.1),
                                range(xgb, :num_round, lower = 300, upper = 700),
								range(xgb, :max_depth, values = [3,4,5])])
 
    m = machine(self_boost_mod, training_data_pca_x, training_data_pca_y) |> fit!

end;
```
"

# ╔═╡ 8a714a8c-7c8b-4dd8-bcc2-598f05564869
md"
### Submission 1 with selected parameters

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
### Submission 2 with selected parameters

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
### Submission 3 with selected parameters

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
### Submission 1 with new features (all params included)

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

# ╔═╡ c62507fd-61d9-4d71-9869-3bb79c642b9a
md"
### Submission 1 with neural network with selected features 

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
### Submission 2 with neural network with selected features 

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
### Submission 4 with neural network with selected features 

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
### Submission 5 with neural network with selected features 

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

# ╔═╡ 803f40f0-d67b-40a9-bd4d-41fe1eacd0e9
md"# Loading and evaluating models"

# ╔═╡ 25123914-bef5-4023-8e1c-e77e555410de


# ╔═╡ 220edda4-f439-49cb-b7a8-f8b3853d8377
begin
	m = machine("machines/16_nn_nf_2.jlso")
	prediction_train = CSV.read(joinpath("predictions/prediction_train_16_nn_nf_2.csv"), DataFrame)
end;

# ╔═╡ 1e442fe7-4fd4-4823-a891-a5a16e5c23ea
begin
	train_x = CSV.read(joinpath("data/train_x.csv"), DataFrame)
	train_y = CSV.read(joinpath("data/train_y.csv"), DataFrame)
	test = CSV.read(joinpath("data/test.csv"), DataFrame);
end;

# ╔═╡ 55c4c4fe-fa9b-476b-a744-1ed319f3e937
plot(m)

# ╔═╡ 5ef01208-2dd1-4fbe-bbad-d913a10cc7a7
#evaluate!(machine(report(m).best_model, select(train_nf_2, Not(:precipitation_nextday)), train_nf_2.precipitation_nextday), measure = AreaUnderCurve())

# ╔═╡ 022ebee0-8d70-47a0-ade5-b50bfcc2b849
fitted_params(m)

# ╔═╡ cec1f5ee-5a3c-4b2b-b605-b7f4b658e7ef
begin
	scatter(prediction_train.p[1:end])
	scatter!(convert(AbstractArray{Float32}, training_data_filled_y)[1:end])

end

# ╔═╡ 4db6ac3a-d187-4a45-aee3-bf3347fcbda7
begin
	sub1 = CSV.read(joinpath("submissions", "submission_16_nn_nf_2.csv"), DataFrame)
	sub2 = CSV.read(joinpath("submissions", "submission_13_nn_nf_2.csv"), DataFrame)

end

# ╔═╡ 22f22845-6033-446c-838c-0ae4410fc2cc
begin
	limits1 = 0
	limits2 = 0
	for i in 1:1200
		if (sub1.precipitation_nextday[i] > 0.4) && (sub1.precipitation_nextday[i] < 0.6)
			limits1 += 1
		end

		if (sub2.precipitation_nextday[i] > 0.4) && (sub2.precipitation_nextday[i] < 0.6)
			limits2 += 1
		end
	end
end

# ╔═╡ 84d9bb72-3337-4141-9b03-e8c1d5e663fd
limits1

# ╔═╡ 87a8f2ea-a922-410d-9150-347a865eced8
limits2

# ╔═╡ f83255f8-7a92-4ec6-bfb4-835b0532b9f0
begin
	scatter(sub1.precipitation_nextday[1:end])
	scatter!(sub2.precipitation_nextday[1:end])
end

# ╔═╡ Cell order:
# ╠═504064e0-4dcb-11ec-380c-8772bf3baad2
# ╟─844fcda0-40ad-4459-937f-560f451d5557
# ╠═f04d9f9a-4f95-4618-b6ba-0881a90b1cf5
# ╠═57269f47-ca0d-4504-841c-2945dc13ca24
# ╠═c11f0fc4-cfee-4303-8456-634dab3c995e
# ╠═6c860d5d-e227-4e9a-b476-0e4aaa35f014
# ╠═617e0349-ffe2-495f-8ac9-6f93d1190a2d
# ╠═1707023e-a0ec-4f16-aa03-eb2760495c9c
# ╠═4b148eef-5521-47c0-8ce0-6354790a24b2
# ╠═591f7943-1b0d-4cec-9cc9-5d846735884f
# ╟─266f5238-1fa0-48a2-b85a-8b5cad2e4bdc
# ╠═d6f8805a-8cbf-4caf-beb0-5622e6eaaf62
# ╠═1851b953-62d2-4dcd-b19b-9b3ecf7dd667
# ╟─0515ad24-ed14-4139-843b-9de7d05622c7
# ╟─a033f5bc-6164-486d-8d10-365671beb4a8
# ╟─cc6d1bfb-e142-4392-86ef-80d9ddd0e6c0
# ╠═c4372b31-7e86-4bb0-abe7-c52eae8436e7
# ╠═ed360f51-9c36-4ed9-925d-ac0d47823b8c
# ╠═76ee4e3d-d8ed-4932-a601-080944290a65
# ╠═150cc1a0-2f60-4ac0-a81a-a439ad199938
# ╠═e4db7640-dd92-4434-92b5-08ac005fff7e
# ╟─d5c0b3f1-d2ad-4131-a1a2-9806d5cbd0f0
# ╟─06407a56-c3b4-4abd-a7f1-efff6b9ba280
# ╟─e65df966-2b3c-4161-b97d-0a432798ab1c
# ╟─2799432a-0ea1-47c3-98fd-52d94402ef51
# ╟─2c659857-3544-4c7d-8de1-bf2ea6af6aaa
# ╠═98b04c3b-3de4-410c-a7eb-c5fa7c495139
# ╠═c4a36662-ad9c-43c2-834b-0e5d23419c58
# ╟─ff637362-88bc-4fa7-83a9-f383a441f514
# ╠═206c3674-4f4b-4d8c-8933-beec8e2a60e4
# ╟─5042fe4c-c6b1-4124-a5e9-887eadd54c93
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
# ╠═ddbfb8aa-8628-4b84-9ba9-06adc09a2b1d
# ╠═fb6e0cd9-0356-4bf4-8599-371753030e48
# ╟─653e837a-5ea0-41e7-a9e9-6f7f283d6acb
# ╟─fc5c6967-734b-4e94-80dc-9dbe5c478c19
# ╟─a42a4137-ff6e-4a61-9a7a-d8dd5a4bc8c6
# ╠═3e4bc22f-f005-4257-90ae-c6fc6cf6f02b
# ╠═c2b766fc-3f93-4b99-96fc-3eb918170463
# ╠═6fd4c29e-1400-47cd-b8e5-82d2520b47df
# ╟─fc1c3dca-d4e5-4a0f-9104-8f00b7c588e9
# ╠═d0bddd3c-b649-452c-a6c3-597c238c6a0f
# ╠═ed4f42af-a0bc-47be-9ea1-9fb30cad0560
# ╟─1cc64552-da14-44c8-8f7c-0ac21977fb7d
# ╠═49217757-5c91-471b-91d9-63f68793807b
# ╠═42db0c60-70d7-472b-b603-3a89b45cdc50
# ╠═246e86fe-b1e9-41ad-8628-700385fd21f7
# ╠═3baf6d35-7277-4457-8faf-ecea6da3639b
# ╠═3856d15e-b094-43a5-9061-d1790b73a48f
# ╠═f7ff1079-2505-44f9-835e-49cb217ac02e
# ╠═75f7e3d5-e095-45bf-a5f6-7126d03554a6
# ╠═424a81a4-2580-481b-81a4-8ffa7ce635f9
# ╠═4b912293-2028-4759-acba-1aa69868ffd7
# ╟─ab18ca30-5e24-486b-9a84-0e37b00b94c1
# ╠═183ea76e-e747-4075-8229-09d191c7e61f
# ╠═e87a6343-5c75-4cdb-bd62-6ca438ce9ba2
# ╟─dffca61c-9eae-45a4-8754-50948979d0ac
# ╟─c4d6535f-ba0c-4300-b5a4-0aeac7b82204
# ╟─703e535a-4ec4-475a-a3fa-e84d88d99d85
# ╠═ca7d7cab-f88c-4aff-9c21-d69116d82727
# ╠═89736a82-f049-4fe3-9946-d8f70434918f
# ╟─da95029c-fbc9-499e-885c-30df96ce8e7e
# ╠═0c5e5a4f-8cf8-47d0-88ca-c875728241b6
# ╠═f0e607c5-1fe3-4da4-a573-68239a11f932
# ╟─0950c80f-f8d3-4326-a3b5-249b1d0488f8
# ╟─404751e7-6a19-4383-81cd-42c943f7a392
# ╠═e6330901-0fad-4de5-859f-1eaf4c0304b8
# ╠═e101a299-6c46-4721-b3d8-7b7028193d22
# ╟─332a087f-983a-49de-90a5-8e33034a3319
# ╟─12886ef5-ae10-4086-9516-48cb4abf455a
# ╟─8ff8d3be-c554-4363-bb20-a36a03b9dad3
# ╠═9a2f0053-34af-4ae1-b0f6-cdf6b6f69ce1
# ╠═3ff1941a-a784-4e7d-94f5-95113a287afb
# ╟─7e5a4c14-ab51-47e1-90b9-cea967f70c89
# ╠═9812e664-cff2-4cc4-b5d9-08af2797c9fd
# ╟─e02938b9-3bcc-4104-a776-3e9d7da03732
# ╠═7ea4ab8e-0b96-4d08-b672-3aee5f6e7fad
# ╠═63044562-86b6-4343-9eae-16e405d20f0e
# ╠═891b3a58-74eb-40c6-b25f-7203f354bd63
# ╠═c79899c5-c88b-4d95-82fb-7578201f7eff
# ╠═10c06962-5d01-4094-a182-b45fbdb65ab7
# ╠═f892effe-2e7f-484e-b212-43a22c66d2bd
# ╠═4f20e66a-d630-4ffa-873a-0329f062244f
# ╠═980c8c10-5cf3-48d7-a93a-4e663f030cec
# ╠═6198fada-af9f-43b1-a6b5-d4ba3132de31
# ╟─efa83958-1350-40c7-b997-814f04ce5042
# ╟─8332735d-30b2-4ba9-a984-e2792d819adf
# ╟─714a02a8-c900-40fd-a734-06e33768ee63
# ╟─84151290-db55-4302-b754-c59f7e389821
# ╟─6408759b-2a69-4dc3-9c13-8de8eb4e96a0
# ╟─6f7aa15a-6028-419a-b7cc-83f3528310d0
# ╟─be56d00c-089c-4e66-91d3-0e70b0e2acd6
# ╟─50bed3df-0584-4aee-8b0c-8448fffe1b1f
# ╟─617dffec-16ec-49f1-a5db-0487d78dfa6d
# ╟─4c39ca91-8553-4d77-9772-d9034b8efd07
# ╟─79a1647e-e34f-4e32-bed7-3c2864717fdb
# ╟─7235988b-2a04-4b4a-ae05-79d3b8faf284
# ╟─8a714a8c-7c8b-4dd8-bcc2-598f05564869
# ╟─490e6e58-5286-4cf4-a758-8e592022aaed
# ╟─3784da1a-3330-4f85-9b5c-f4722da1eebb
# ╟─41788e47-3df4-4aae-acdd-48240ecf5a88
# ╟─c62507fd-61d9-4d71-9869-3bb79c642b9a
# ╟─720c2f82-2d7f-4dca-af12-a09f41ae2fe2
# ╟─051adb23-2a96-409d-96c3-9cc60d377dba
# ╟─7f98dc7e-c934-4eea-9ba3-b9d8d9cdda02
# ╟─803f40f0-d67b-40a9-bd4d-41fe1eacd0e9
# ╠═25123914-bef5-4023-8e1c-e77e555410de
# ╠═220edda4-f439-49cb-b7a8-f8b3853d8377
# ╠═1e442fe7-4fd4-4823-a891-a5a16e5c23ea
# ╠═55c4c4fe-fa9b-476b-a744-1ed319f3e937
# ╠═5ef01208-2dd1-4fbe-bbad-d913a10cc7a7
# ╠═022ebee0-8d70-47a0-ade5-b50bfcc2b849
# ╠═cec1f5ee-5a3c-4b2b-b605-b7f4b658e7ef
# ╠═4db6ac3a-d187-4a45-aee3-bf3347fcbda7
# ╠═22f22845-6033-446c-838c-0ae4410fc2cc
# ╠═84d9bb72-3337-4141-9b03-e8c1d5e663fd
# ╠═87a8f2ea-a922-410d-9150-347a865eced8
# ╠═f83255f8-7a92-4ec6-bfb4-835b0532b9f0
# ╟─afdb8fbe-0f01-4691-a5b9-9a715c29d5ea
