# Initialize the environment
using Pkg

Pkg.activate(".")
Pkg.instantiate()
ENV["DATADEPS_ALWAYS_ACCEPT"] = true # accept the download of the MNIST dataset
SHOW_PROGRESS_BAR = true # turn off progress bars

using DelimitedFiles

dim_input=18 #input features here are 8-dimensional vectors
dim_output=1 #outcomes are two-dimensional vectors in this example
subnum=4 #2
actfile=4
recs = 200
#recs = 50
version = 3 #5
data_dir = "./DATA/"

#Pkg.build()
CUDA_VISIBLE_DEVICES=""
using Flux

using Bloqade

# import required libraries
using MultivariateStats
using OneHotArrays
using ProgressBars
using JLD2
using Statistics
using DataFrames
using Plots
using CSV
using StatisticalMeasures

# Define the struct
Base.@kwdef struct DetuningLayer
    atoms # atom positions
    readouts # readout observables
    Ω::Float64 # Rabi frequency
    t_start::Float64 # evolution starting time
    t_end::Float64 # evolution ending time
    step::Float64  # readout time step
    reg::AbstractArrayReg # quantum state storage
end

# implement functions that apply a `DetuningLayer` to a matrix containing scaled detunings 
#for each image
function (layer::DetuningLayer)(x::Matrix{<:Real})
    iter = SHOW_PROGRESS_BAR ? ProgressBar(1:size(x, 2)) : 1:size(x, 2)
    outs = [layer(x[:, i][:]) for i in iter]
    return hcat(outs...)
end

# Iterate over each single image
# For more details on simulation, please refer to Bloqade.jl and Yao.jl
function (layer::DetuningLayer)(x::Vector{<:Real})
    # define hamiltonian, detunings parameterized in terms of pca values (x)  
    h = rydberg_h(layer.atoms; Δ = x, Ω = layer.Ω) 
    
    # system starts in zero state
    reg = layer.reg
    set_zero_state!(reg) 
    
    t_start = layer.t_start
    t_end = layer.t_end
    t_step = layer.step
    
    # initialize output vector
    steps = floor(Int, (t_end - t_start) / t_step)
    out = zeros(steps * length(layer.readouts))
    
    # Simulate the quantum evolution with Krylov methods and store the readouts
    i = 1
    prob =  KrylovEvolution(reg, layer.t_start:layer.step:layer.t_end, h)
    for (step, reg, _) in prob # step through the state at each time step 
        step == 1 && continue # ignore first time step, this is just the initial state
        for op in layer.readouts
            out[i] = real(expect(op, reg)) # store the expectation of each operator for the given state in the output vector 
            i+=1
        end
    end
    return out
end

function train8(xs_train, ys_train, xs_test, ys_test; 
    regularization::Float64 = 0.0, nepochs::Int = 100, batchsize::Int = 100, 
    opt = Flux.Adam(0.01), verbose::Bool, nonlinear::Bool=false)
    dout=length(ys_train[:,1])
    model_at = Chain(
    Dense(length(xs_train[:, 1]), dout)
    )

    if nonlinear
        model_at = Chain(
            Dense(length(xs_train[:, 1]), 100, relu),
            Dense(100, 100, relu),
            Dense(100, dout)
        )
    end

    loader = Flux.DataLoader((data = xs_train, label = ys_train); batchsize, shuffle=true);
    #ps = Flux.trainable(model_at)
    opt_state = Flux.setup(opt, model_at)
    #ys = zeros(length(xs_test),1)
    
    verbose && println("Training...")
    losses = zeros(nepochs)
    losses_train = zeros(nepochs)
    losses_test = zeros(nepochs)
    mae_losses_train = zeros(nepochs)
    mae_losses_test = zeros(nepochs)
    for epoch in (verbose ? ProgressBar(1:nepochs) : 1:nepochs)
        l = 1.0
        for (x, y) in loader
            if iszero(regularization)
                grads = Flux.gradient(model_at) do m
                    ŷ = m(x)
                    l = Flux.mse(ŷ, y)
                end
            else
                grads = Flux.gradient(model_at) do
                    ŷ = m(x)
                    ps = Flux.trainables(m)
                    l = Flux.mse(ŷ, y) + regularization * sum(sum(abs, p) for p in ps)
                end
            end
            Flux.update!(opt_state, model_at, grads[1])
        end
    losses_train[epoch] = Flux.mse(model_at(xs_train), ys_train)
    losses_test[epoch] = Flux.mse(model_at(xs_test), ys_test)
    mae_losses_train[epoch] = Flux.mae(model_at(xs_train), ys_train)
    mae_losses_test[epoch] = Flux.mae(model_at(xs_test), ys_test)
    end
    y_pred = model_at(xs_test)
    return losses_train, losses_test, y_pred, model_at, mae_losses_train, mae_losses_test
end

function qrc_reg(subnum, actfile, recs)
    println("in qrc_reg")

    df_xlsx_train_X = DataFrame(CSV.File(data_dir*"X_train_"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv"))
    df_xlsx_test_X = DataFrame(CSV.File(data_dir*"X_test_"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv"))
    df_xlsx_train_y = DataFrame(CSV.File(data_dir*"y_train_"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv"))
    df_xlsx_test_y = DataFrame(CSV.File(data_dir*"y_test_"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv"))
    println("size of train_X is", size(df_xlsx_train_X))
    
    names(df_xlsx_test_X)
    
    train_data = DataFrames.select(df_xlsx_train_X, Not(:Column1, :cluspred)) #, :f_diesel, :f_petrol, :f_other, :tran_auto, :tran_manual, :brand_hyundi, :brand_vwaudi, :brand_toyota, :brand_ford, :brand_mercbwm, :brand_other])))
    test_data = DataFrames.select(df_xlsx_test_X, Not(:Column1, :cluspred)) #, :f_diesel, :f_petrol, :f_other, :tran_auto, :tran_manual, :brand_hyundi, :brand_vwaudi, :brand_toyota, :brand_ford, :brand_mercbwm, :brand_other])))
    train_outcomes = DataFrames.select(df_xlsx_train_y, Not(:Column1))
    test_outcomes = DataFrames.select(df_xlsx_test_y, Not(:Column1))

    #DEBUG
    #train_data = train_data[:, 1:dim_input]
    #test_data = test_data[:, 1:dim_input]
    println("size of train_data is", size(train_data))
    println("size of test_data is", size(test_data))
    println("size of train_outcomes is", size(train_outcomes))
    println("size of test_outcomes is", size(test_outcomes))
    
    
    
    #Inverting matrix and converting all to Float64
    train_inv = Matrix(train_data)' #similar(train_data', Float64)
    test_inv = Matrix(test_data)' #similar(test_data', Float64)
    
    xs = train_inv
    
    Δ_max = 6.0
    spectral = max(abs(maximum(train_inv)), abs(minimum(train_inv)))
    
    xs = train_inv/spectral * Δ_max # to make sure values to be between [-6.0, 6.0]
    

    
    # Generate atom positions for the toy model
    atoms = generate_sites(ChainLattice(), dim_input; scale = 10); # put atoms in a chain with 9 micron spacing
    
    # create all single site Zᵢ and correlator ZᵢZⱼ readouts 
    nsites = length(atoms)
    readouts = AbstractBlock[put(nsites, i => Z) for i in 1:nsites]
    for i in 1:nsites
        for j in i+1:nsites
            push!(readouts, chain(put(nsites, i => Z), put(nsites, j => Z)))
        end
    end
    
    # build preprocessing layer 
    pre_layer = DetuningLayer(;
        atoms, 
        readouts, 
        Ω = 2π, 
        t_start = 0.0, 
        t_end = 4.3, #best4.3, #default 4.0, 
        step = 0.4, #best 0.4, #default 0.5, 
        reg = zero_state(nsites)
    );
    
    # uncomment the next line to see progress bar
    SHOW_PROGRESS_BAR = false
    println("size of xs is", size(xs))
    embeddings = pre_layer(xs)
    
    any(isnan.(test_inv))
    
    test_features_qrc = test_inv/spectral * Δ_max 
    any(isnan.(test_features_qrc))
    
    # quantum embeddings for 100 test samples
    println("before pre_layer")
    test_embeddings = pre_layer(test_features_qrc)
    println("before training 1")

    
    writedlm(data_dir*"records"*"$recs"*"/merck_train_embeddings_recs"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv",  embeddings, ',')
    writedlm(data_dir*"records"*"$recs"*"/merck_test_embeddings_recs"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv",  test_embeddings, ',')
    writedlm(data_dir*"records"*"$recs"*"/merck_train_outcomes_recs"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv",  Matrix(train_outcomes), ',')
    writedlm(data_dir*"records"*"$recs"*"/merck_test_outcomes_recs"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv",  Matrix(test_outcomes), ',')
    
    train_outcomes_inv = Matrix(permutedims(train_outcomes))
    test_outcomes_inv = Matrix(permutedims(test_outcomes))
    println("before training 2")
    mse_train, mse_test, y_pred, model_at, mae_train, mae_test = train8(embeddings, train_outcomes_inv, test_embeddings, test_outcomes_inv,
     regularization= 0.00, nepochs= 10, batchsize= 2, opt = Flux.Adam(0.0038), verbose=false)
    # nepochs = 1000
    println("Train MSE QRC=",minimum(mse_train)) #
    println("Test MSE QRC=", minimum(mse_test))
    Plots.plot(mse_test, label="Train MSE") 
    plot!(mse_test, label="Test MSE")
    
    println("Train MAE QRC=",minimum(mae_train)) #Robust to Outliers
    println("Test MAE QRC=", minimum(mae_test))
    Plots.plot(mae_test, label="Train MSE")
    plot!(mae_test, label="Test MSE")
    
    y_pred64 = convert(Array{Float64}, y_pred)
    
    test_outcomes_trans = transpose(Matrix(test_outcomes))
    test_outcomes_inv = Matrix(test_outcomes)'
    
    ## R2 calculation
    f_prediction = vec(y_pred)
    y_target = vec(test_outcomes_inv)
    
    MSE_y_f = mean((y_target .- f_prediction).^2)
    MSE_y_y_bar = mean((y_target .- mean(y_target)).^2)
    R2 = 1-MSE_y_f/MSE_y_y_bar
    
    rmse = StatisticalMeasures.rmse(vec(y_pred64), vec(test_outcomes_inv)) 
    mape = StatisticalMeasures.mape(vec(y_pred64), vec(test_outcomes_inv))
    
    println("Test R2 Explained Variance QRC=",R2) #Explained variance
    println("Test Root Mean Square Error QRC=",rmse) #sensitive to outliers, scaled to to target variable 
    println("Test Mean Absolute Percent Error QRC=",mape) #Mean Absolute percent Error - Percent equivalent to MAE
    
    train_data_mat = Matrix(permutedims(train_data))
    train_outcomes_mat = Matrix(permutedims(train_outcomes))
    test_data_mat = Matrix(permutedims(test_data))
    test_outcomes_mat = Matrix(permutedims(test_outcomes))
    
    println(size(train_data_mat))
    println(size(train_outcomes_mat))
    println(size(test_data_mat))
    println(size(test_outcomes_mat))
    
    loss_train_lin, loss_test_lin, y_pred_lin, mod_lin, mae_train_lin, mae_test_lin  = train8(train_data_mat, train_outcomes_mat, test_data_mat, test_outcomes_mat,
     regularization= 0.00, nepochs= 1000, batchsize= 5, opt = Flux.Adam(0.0038), verbose=false, nonlinear=false)
    
    println("Train MSE linear=", minimum(loss_train_lin))
    println("Test MSE linear=", minimum(loss_test_lin))
    Plots.plot(loss_train_lin, label="Train MSE")
    plot!(loss_test_lin, label="Test MSE")
    
    println("Train MAE QRC=",minimum(mae_train_lin)) #Robust to Outliers
    println("Test MAE QRC=", minimum(mae_test_lin))
    Plots.plot(mae_test_lin, label="Train MSE")
    plot!(mae_test_lin, label="Test MSE")
    
    ## R2 calculation
    y_pred64_lin = convert(Array{Float64}, y_pred_lin)
    test_outcomes_inv = Matrix(test_outcomes)'
    
    
    f_prediction = vec(y_pred64_lin)
    y_target = vec(test_outcomes_inv)
    
    MSE_y_f = mean((y_target .- f_prediction).^2)
    MSE_y_y_bar = mean((y_target .- mean(y_target)).^2)
    R2 = 1-MSE_y_f/MSE_y_y_bar
    
    
    y_pred64_lin
    
    test_outcomes_trans = transpose(Matrix(test_outcomes))
    
    ## R2 calculation
    f_prediction = vec(y_pred64_lin)
    #y_target = vec(test_outcomes_inv)
    
    MSE_y_f = mean((y_target .- f_prediction).^2)
    MSE_y_y_bar = mean((y_target .- mean(y_target)).^2)
    R2 = 1-MSE_y_f/MSE_y_y_bar
    
    rmse = StatisticalMeasures.rmse(vec(y_pred64_lin), vec(test_outcomes_inv)) 
    mape = StatisticalMeasures.mape(vec(y_pred64_lin), vec(test_outcomes_inv))
    
    println("Test R2 Explained Variance QRC=",R2) #Explained variance
    println("Test Root Mean Square Error QRC=",rmse) #sensitive to outliers, scaled to to target variable 
    println("Test Mean Absolute Percent Error QRC=",mape) #Mean Absolute percent Error - Percent equivalent to MAE
    
    # Generate atom positions for the toy model
    atoms = generate_sites(ChainLattice(), dim_input; scale = 10); # put atoms in a chain with 9 micron spacing
    
    # create all single site Zᵢ and correlator ZᵢZⱼ readouts 
    nsites = length(atoms)
    readouts = AbstractBlock[put(nsites, i => Z) for i in 1:nsites]
    #for i in 1:nsites
    #    for j in i+1:nsites
    #        push!(readouts, chain(put(nsites, i => Z), put(nsites, j => Z)))
    #    end
    #end
    
    # build preprocessing layer 
    pre_layer_lin = DetuningLayer(;
        atoms, 
        readouts, 
        Ω = 2π, 
        t_start = 0.0, 
        t_end = 4.3, #best4.3, #default 4.0, 
        step = 0.4, #best 0.4, #default 0.5, 
        reg = zero_state(nsites)
    );
    
    # uncomment the next line to see progress bar
    SHOW_PROGRESS_BAR = false
    lin_embeddings = pre_layer_lin(xs)
    
    # quantum embeddings for 100 test samples
    lin_test_embeddings = pre_layer_lin(test_features_qrc)
    
    #Original values
    #loss_train, loss_test= train(embeddings, train_outcomes', test_embeddings, test_outcomes',
    # regularization= 0.00, nepochs= 1000, batchsize= 5, opt = Flux.Adam(0.0038), verbose=false)
    #, mae_train, mae_test, ce_train, ce_test 
    mse_train_expr, mse_test_expr, y_pred_expr, model_at_expr, mae_train_expr, mae_test_expr = train8(lin_embeddings, train_outcomes_inv, lin_test_embeddings, test_outcomes_inv,
     regularization= 0.00, nepochs= 1000, batchsize= 2, opt = Flux.Adam(0.0038), verbose=false)
    
    #loss_train, loss_test= train(embeddings, train_outcomes, test_embeddings, test_outcomes,
    # regularization= 0.00, nepochs= 1000, batchsize= 5, opt = Flux.Adam(0.0038), verbose=false)
    
    println("Train MSE QRC=",minimum(mse_train_expr)) #
    println("Test MSE QRC=", minimum(mse_test_expr))
    Plots.plot(mse_test_expr, label="Train MSE") 
    plot!(mse_test_expr, label="Test MSE")
    
    println("Train MAE QRC=",minimum(mae_train_expr)) #Robust to Outliers
    println("Test MAE QRC=", minimum(mae_test_expr))
    Plots.plot(mae_test_expr, label="Train MSE")
    plot!(mae_test_expr, label="Test MSE")
    
    y_pred64_expr = convert(Array{Float64}, y_pred_expr)
    
    f_prediction = vec(y_pred64_expr)
    y_target = vec(test_outcomes_inv)
    
    MSE_y_f = mean((y_target .- f_prediction).^2)
    MSE_y_y_bar = mean((y_target .- mean(y_target)).^2)
    R2 = 1-MSE_y_f/MSE_y_y_bar
    
    #r2 = rsquared(vec(y_pred64_expr), vec(test_outcomes_inv)) 
    rmse_expr = StatisticalMeasures.rmse(vec(y_pred64_expr), vec(test_outcomes_inv)) 
    mape_expr = StatisticalMeasures.mape(vec(y_pred64_expr), vec(test_outcomes_inv))
    
    writedlm(data_dir*"records"*"$recs"*"/merck_train_emb_lin_rec"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv",  lin_embeddings, ',')
    writedlm(data_dir*"records"*"$recs"*"/merck_test_emb_lin_rec"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv",  lin_test_embeddings, ',')
    
    writedlm(data_dir*"records"*"$recs"*"/merck_train_outcomes_lin_rec"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv",  Matrix(train_outcomes), ',')
    writedlm(data_dir*"records"*"$recs"*"/merck_test_outcomes_lin_rec"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv",  Matrix(test_outcomes), ',')
    
    writedlm(data_dir*"records"*"$recs"*"/merck_y_predictions_rec"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv", y_pred64_lin)
    writedlm(data_dir*"records"*"$recs"*"/merck_y_predicitions_lin_rec"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv", y_pred64_expr)
    writedlm(data_dir*"records"*"$recs"*"/merck_y_predicitions_pca_rec"*"$recs"*"rec_sub"*"$subnum"*"act"*"$actfile"*"v"*"$version"*".csv", y_pred_lin)
end

#Actfile 4
#subnum=1
#actfile=4

# acts = [5,14,15,9]
# subs = [1,2,3,4,5]
# recs = [100,200] 


#acts = [15, 14, 9, 5] #,14,15,9]
acts = [4]
subs = [1,2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
#recs = [100,200] 
recs = [200] 
version = 3

for i in acts
    for j in subs
        for k in recs
            println("Actfile=",i)
            println("Subsample=",j)
            println("Record Numbers=",k)
            qrc_reg(j, i, k)
        end
    end
end
#subnum=2
#qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# actfile=4
# subnum=1
# recs = 100

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# #Actfile 2
# actfile=2
# subnum=1
# recs = 100

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# subnum=1
# #actfile=2
# #recs = 200

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# #Actfile 3
# actfile=3
# subnum=1
# recs = 100

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# subnum=1
# #recs = 200

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# #Actfile 3
# actfile=5
# subnum=1
# recs = 100

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# subnum=1
# #recs = 200

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# #Actfile 6
# actfile=6
# subnum=1
# recs = 100

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)

# subnum=1
# #recs = 200

# qrc_reg(subnum, actfile, recs)

# subnum=2
# qrc_reg(subnum, actfile, recs)

# subnum=3
# qrc_reg(subnum, actfile, recs)

# subnum=4
# qrc_reg(subnum, actfile, recs)

# subnum=5
# qrc_reg(subnum, actfile, recs)