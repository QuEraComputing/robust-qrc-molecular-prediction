# Step 1: Configure PyCall to use Conda
#Server
#ENV["PYTHON"] = "/home/shared/miniconda3"
#Local
#ENV["PYTHON"] = "/home/shared/miniconda3/envs/rapids-25.02/bin/python"
#ENV["PYTHON"] = "/home/shared/miniconda3/envs/qrcjulia/bin/python"
using Pkg
# using PyCall
# Pkg.build("PyCall")
Pkg.activate(".")
Pkg.instantiate()
ENV["DATADEPS_ALWAYS_ACCEPT"] = true # accept the download of the MNIST dataset
SHOW_PROGRESS_BAR = true # turn off progress bars

# # Step 3: Import RandomForestRegressor
# using PyCall
# sklearn = pyimport("sklearn")
# RandomForestRegressor = sklearn.ensemble.RandomForestRegressor


# Step 2: Install sklearn using Conda
#using Conda
#Conda.add("scikit-learn")

# Step 3: Import RandomForestRegressor

using Bloqade

using CSV
using DataFrames
using Random
using Plots
using Flux
using ProgressMeter
#using ScikitLearn
using LinearAlgebra 
 # Import the LinearAlgebra package to use the norm function
#@sk_import ensemble: RandomForestRegressor
using Statistics  # Import the Statistics module to use the mean function
using OneHotArrays
using MultivariateStats  # Import the MultivariateStats package for PCA
import MultivariateStats: fit  # Import the fit function from MultivariateStats
using DifferentialEquations
using MLDatasets
using DelimitedFiles

# using ScikitLearn
# @sk_import ensemble: RandomForestRegressor

# # Import RandomForestRegressor from sklearn
# sklearn = pyimport("sklearn")
# RandomForestRegressor = sklearn.ensemble.RandomForestRegressor

# Define the struct for CRC layer
Base.@kwdef struct ClassicalDetuningLayer
    atoms # atom positions
    Ω::Real # Rabi frequency
    t_start::Real # evolution starting time
    t_end::Real # evolution ending time
    step::Real  # readout time step
    state::Vector{<:Real} # classical state
    readout::String #readout ("s" or "ss")
end

# Function to generate Vmat
function generate_Vmat(locs::Vector{Vector{Float64}}, C6=862690*2π)
    nsites = length(locs)
    Vmat = zeros(nsites, nsites)
    VDel = zeros(nsites)
    for i in 1:nsites
        for j in 1:nsites
            if i != j
                Vmat[i, j] = C6 / (norm(locs[i, :] - locs[j, :]))^6
                VDel[i] += Vmat[i, j]
            end
        end
    end
    return Vmat, VDel
end

# Function to calculate derivatives
function deriv!(du, u, p::Tuple{Int, Vector{<:Real}, <:Real, Matrix{<:Real}, Vector{<:Real}}, t)
    nsites, Delta, Omega, Vmat, VDel = p
    Bv = Vmat * u[3:3:3*nsites]
    for i in 1:nsites
        Bi = [Omega, 0, -Delta[i] + Bv[i] / 2 + VDel[i] / 2] / 2
        s = u[3*i-2:3*i]
        du[3*(i-1)+1] = Bi[3] * s[2] - Bi[2] * s[3]
        du[3*(i-1)+2] = Bi[1] * s[3] - Bi[3] * s[1]
        du[3*(i-1)+3] = Bi[2] * s[1] - Bi[1] * s[2]
    end
    return du
end

# Function to apply classical layer
function apply_classical_layer(layer::ClassicalDetuningLayer, x::Vector{<:Real})
    locs = layer.atoms
    readout = layer.readout
    nsites = length(locs)
    Vmat, VDel = generate_Vmat(locs)
    u0 = layer.state

    t_start = layer.t_start
    t_end = layer.t_end
    t_step = layer.step

    steps = floor(Int, (t_end - t_start) / t_step)
    out2 = zeros(3*nsites, 3*nsites, steps)
    out3 = zeros(nsites, nsites, steps)
    out31 = zeros(div(nsites*(nsites-1), 2), steps)

    Omega = layer.Ω
    timespan = (t_start, t_end)
    prob = ODEProblem(deriv!, u0, timespan, (nsites, x, Omega, Vmat, VDel))
    sol = solve(prob, RK4(), saveat=t_step, adaptive=false, dt=1e-3, save_start=false)
    sol3 = sol[3:3:3*nsites, :]

    if readout == "s"
        out = reduce(vcat, sol)
    elseif readout == "ss"
        for i in 1:steps
            out2[:, :, i] = sol[:, i] .* sol[:, i]'
        end
        out = reduce(vcat, (reduce(vcat, sol), reduce(vcat, out2)))
    elseif readout == "zz"
        for i in 1:steps
            ind = 1
            for i1 in 1:nsites
                for i2 in (i1+1):nsites
                    out31[ind, i] = sol3[i1, i] * sol3[i2, i]
                    ind += 1
                end
            end
        end
        out = reduce(vcat, (reduce(vcat, sol3), reduce(vcat, out31)))
    elseif readout == "szz"
        ind = 1
        for i in 1:steps
            for i1 in 1:nsites
                for i2 in (i1+1):nsites
                    out31[ind, i] = sol3[i1, i] * sol3[i2, i]
                    ind += 1
                end
            end
        end
        out = reduce(vcat, (reduce(vcat, sol), reduce(vcat, out3)))
    elseif readout == "z"
        out = reduce(vcat, sol3)
    end

    return out
end

# Function to apply classical layer to a matrix
function apply_classical_layer(layer::ClassicalDetuningLayer, x::Matrix{<:Real})
    iter = SHOW_PROGRESS_BAR ? ProgressBar(1:size(x, 2)) : 1:size(x, 2)
    outs = [apply_classical_layer(layer, x[:, i][:]) for i in iter]
    return hcat(outs...)
end

accuracy(model, xs, targets) = sum(onecold(model(xs), 0:9) .== targets)/length(targets)
function train_linear_nn!(xs_train, ys_train, xs_test, ys_test; 
    regularization::Float64 = 0.0, nepochs::Int = 100, batchsize::Int = 100, 
    opt = Flux.Adam(0.01), verbose::Bool, nonlinear::Bool=false)
    
    model = Chain(
    Dense(length(xs_train[:, 1]), 10),
    softmax
    )

    if nonlinear
        model = Chain(
            Dense(length(xs_train[:, 1]), 100, relu),
            Dense(100, 100, relu),
            Dense(100, 10),
            softmax
        )
    end

    loader = Flux.DataLoader((data = xs_train, label = ys_train); batchsize, shuffle=true);
    ps = Flux.params(model)

    verbose && println("Training...")
    losses = zeros(nepochs)
    accs_train = zeros(nepochs)
    accs_test = zeros(nepochs)
    for epoch in (verbose ? ProgressBar(1:nepochs) : 1:nepochs)
    l = 1.0
    for (x, y) in loader
        grads = Flux.gradient(ps) do
            ŷ = model(x)
            if iszero(regularization)
                l = Flux.crossentropy(ŷ, y)
            else
                l = Flux.crossentropy(ŷ, y) + regularization * sum(sum(abs, p) for p in ps)
            end
        end
        Flux.update!(opt, ps, grads)
    end
    losses[epoch] = Flux.crossentropy(model(xs_train), ys_train)
    accs_train[epoch] = accuracy(model, xs_train, onecold(ys_train, 0:9))
    accs_test[epoch] = accuracy(model, xs_test, ys_test)
    end
    return losses, accs_train, accs_test
end

function train_svm(xs, ys, test_features, test_targets)
    model = svmtrain(xs, onecold(ys, 0:9); svmtype = SVC, kernel = Kernel.Linear)
    train_ŷ, train_decision_values = svmpredict(model, xs);
    acc_train = mean(train_ŷ .== onecold(ys, 0:9))
    ŷ, decision_values = svmpredict(model, test_features);
    acc_test = mean(ŷ .== test_targets)
    return acc_train, acc_test
end


# Function to train and evaluate RandomForestRegressor
function train_random_forest(xs_train, ys_train, xs_test, ys_test)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(xs_train, ys_train)
    y_train_pred = model.predict(xs_train)
    y_test_pred = model.predict(xs_test)
    train_mse = mean((y_train_pred .- ys_train).^2)
    test_mse = mean((y_test_pred .- ys_test).^2)
    return train_mse, test_mse
end

# Generate synthetic data for demonstration
function generate_synthetic_data(n_samples::Int, n_features::Int)
    xs = randn(n_samples, n_features)
    ys = sum(xs, dims=2) + 0.1 * randn(n_samples)  # Simple linear relationship with noise
    return xs, ys
end

version = [5]
# original version = [ 3]
all_results_df = DataFrame()

acts = [14] #5,9, 14,15] #  4]#,]
subs = [1,2,3,4,5] #,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
#recs = [100,200,800]#, 200, 800] 
recs = [100]
#acts = 4
#subs = 1
#recs = 100 
SHOW_PROGRESS_BAR = false 

for act in acts
    for sub in subs
        for rec in recs
            for ver in version
			
                println("Processing SHAP Classical act: $act, sub: $sub, rec: $rec")
                
                # Read the data
                df_train = CSV.read("./X_train_$(rec)rec_sub$(sub)act$(act)v$(ver).csv", DataFrame)
                df_test = CSV.read("./X_test_$(rec)rec_sub$(sub)act$(act)v$(ver).csv", DataFrame)
                df_ytrn = CSV.read("./y_train_$(rec)rec_sub$(sub)act$(act)v$(ver).csv", DataFrame, select=["Act"])
                df_ytst = CSV.read("./y_test_$(rec)rec_sub$(sub)act$(act)v$(ver).csv", DataFrame, select=["Act"])
    
                # Convert DataFrames to Arrays for sklearn
                # xs_train = Matrix(df_train)
                # ys_train = Vector(df_ytrn[:, "Act"])
                # xs_test = Matrix(df_test)
                # ys_test = Vector(df_ytst[:, "Act"])
    
                train_data = DataFrames.select(df_train, Not(:Column1, :cluspred)) #, :f_diesel, :f_petrol, :f_other, :tran_auto, :tran_manual, :brand_hyundi, :brand_vwaudi, :brand_toyota, :brand_ford, :brand_mercbwm, :brand_other])))
                test_data = DataFrames.select(df_test, Not(:Column1, :cluspred)) #, :f_diesel, :f_petrol, :f_other, :tran_auto, :tran_manual, :brand_hyundi, :brand_vwaudi, :brand_toyota, :brand_ford, :brand_mercbwm, :brand_other])))
                train_outcomes = DataFrames.select(df_ytrn, [:Act])
                test_outcomes = DataFrames.select(df_ytst, [:Act])
                size(train_data)
                
    			# Check dimensions
                # if size(xs_train, 2) != size(xs_test, 2)
                #     println("Dimension mismatch: xs_train has $(size(xs_train, 2)) features, xs_test has $(size(xs_test, 2)) features")
                #     continue
                # end
    
    			train_inv = Matrix(train_data)' #similar(train_data', Float64)
                test_inv = Matrix(test_data)' #similar(test_data', Float64)
    
    			# We first use PCA to downsample the data into 10-dimensional vectors
    			dim_pca = 18
    
    			# # Use the `fit` function from the `MultivariateStats` package to generate the projection operator for PCA
    			#             model_pcatrn = fit(PCA, xs_train; maxoutdim=dim_pca)
    			#             xs_train_pca = MultivariateStats.transform(model_pcatrn, xs_train)
    			#
    			# model_pcatst = fit(PCA, xs_test; maxoutdim=dim_pca)
    			# xs_test_pca = MultivariateStats.transform(model_pcatst, xs_test)
    
                # Normalize features
                Δ_max = 6
                spectral_trn = max(abs(maximum(train_inv)), abs(minimum(train_inv)))
                xs_train_pca = train_inv / spectral_trn * Δ_max
                spectral_tst = max(abs(maximum(test_inv)), abs(minimum(test_inv)))
                xs_test_pca = test_inv / spectral_tst * Δ_max
    
    			d = 10.0
    			nsites=dim_pca
    			locs = [[i*d,0.0] for i in 1:nsites] # put atoms in a chain with 10 micron spacing
    			u0 =zeros(3*nsites)
    			for i in 1:nsites
    			    u0[3*i] = -1 
    			end
    			pre_layer = ClassicalDetuningLayer(;
    			    atoms=locs, 
    			    Ω = 2π, 
    			    t_start = 0.0, 
    			    t_end = 4.0, 
    			    step = 0.5, 
    			    state=u0,
    			    readout="zz"
    			);
    			
    			# uncomment the next line to see progress bar
    			SHOW_PROGRESS_BAR = false 
                embeddings_train = apply_classical_layer(pre_layer, xs_train_pca)
                embeddings_test = apply_classical_layer(pre_layer, xs_test_pca)
    			
                #xs_train = Matrix(df_train)
                ys_train = Vector(train_outcomes[:, "Act"])
                #xs_test = Matrix(df_test)
                ys_test = Vector(test_outcomes[:, "Act"])
    
                writedlm("records"*"$rec"*"/merck_train_embbedding_rec"*"$rec"*"rec_sub"*"$sub"*"act"*"$act"*"v"*"$ver"*"_crc"*".csv",    embeddings_train, ',')
                writedlm("records"*"$rec"*"/merck_test_embbedding_rec"*"$rec"*"rec_sub"*"$sub"*"act"*"$act"*"v"*"$ver"*"_crc"*".csv",  embeddings_test, ',')
                
                # writedlm("records"*"$rec"*"/merck_train_outcomes_lin_rec"*"$rec"*"rec_sub"*"$subs"*"act"*"$act"*"v"*"$version"*"_crc_"*".csv",  Matrix(ys_train), ',')
                # writedlm("records"*"$rec"*"/merck_test_outcomes_lin_rec"*"$rec"*"rec_sub"*"$subs"*"act"*"$act"*"v"*"$version"*"_crc_"*".csv",  Matrix(ys_test), ',')   
                writedlm("records"*"$rec"*"/merck_train_outcomes_emb_rec"*"$rec"*"rec_sub"*"$sub"*"act"*"$act"*"v"*"$ver"*"_crc"*".csv", reshape(ys_train, :, 1), ',')
    writedlm("records"*"$rec"*"/merck_test_outcomes_emb_rec"*"$rec"*"rec_sub"*"$sub"*"act"*"$act"*"v"*"$ver"*"_crc"*".csv", reshape(ys_test, :, 1), ',')
                
                # Train and evaluate RandomForestRegressor
                # train_mse, test_mse = train_random_forest(embeddings_train', ys_train, embeddings_test', ys_test)
                # results_dict_qrc = Dict("train_mse" => train_mse, "test_mse" => test_mse)
    			
                # # train_outcomes_inv = Matrix(permutedims(train_outcomes))
                # # test_outcomes_inv = Matrix(permutedims(test_outcomes))
                # # # Train and evaluate RandomForestRegressor
                # # train_mse, test_mse = train_random_forest(embeddings_train', train_outcomes, embeddings_test', test_outcomes)
                # # results_dict_qrc = Dict("train_mse" => train_mse, "test_mse" => test_mse)
                
                # # print(size(embeddings_train))
                # # print(size(embeddings_test))
                # # print(size(train_outcomes_inv))
                # # print(size(test_outcomes_inv))
    
                # print(size(embeddings_train'))
                # print(size(embeddings_test'))
                # print(size(train_outcomes))
                # print(size(test_outcomes))
    
                # # Store results
                # results_df_qrc = DataFrame(results_dict_qrc)
                # insertcols!(results_df_qrc, :act => act)
                # insertcols!(results_df_qrc, :sub => sub)
                # insertcols!(results_df_qrc, :rec => rec)
                # insertcols!(results_df_qrc, :source => "results_dict_qrc")
                # all_results_df = vcat(all_results_df, results_df_qrc)
            end
        end
    end
end

println("Finished act14 Run for 100,200,800")
# Function to calculate mean squared error
# mse(y_pred, y_true) = mean((y_pred .- y_true).^2)
# Output all_results_df to a CSV file
#CSV.write("crc_all_results_randfor100-200-800_act14.csv", all_results_df)

#
#
# # Example usage with synthetic data
# n_samples = 1000
# n_features = 10
# xs, ys = generate_synthetic_data(n_samples, n_features)
#
# # Split the data into training and testing sets
# train_ratio = 0.8
# n_train = floor(Int, train_ratio * n_samples)
# xs_train = xs[1:n_train, :]
# ys_train = ys[1:n_train]
# xs_test = xs[n_train+1:end, :]
# ys_test = ys[n_train+1:end]
#
# # Train the RandomForestRegressor and evaluate
# train_mse, test_mse = train_random_forest(xs_train, ys_train, xs_test, ys_test)
# println("Train MSE: $train_mse")
# println("Test MSE: $test_mse")