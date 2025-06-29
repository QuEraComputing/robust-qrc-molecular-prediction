using Pkg

Pkg.activate(".")
Pkg.instantiate()
ENV["DATADEPS_ALWAYS_ACCEPT"] = true # accept the download of the MNIST dataset
SHOW_PROGRESS_BAR = true # turn off progress bars

using CSV
using DataFrames
using Random
using Bloqade
using BloqadeNoisy

using DelimitedFiles
using Flux
using MultivariateStats
using OneHotArrays
using ProgressBars
using JLD2
using Statistics
using Plots
using StatisticalMeasures

subnum = 1
dim_input = 18 # input features here are 10-dimensional vectors
dim_output = 1 # outcomes are two-dimensional vectors in this example
actfile = 14
recs = 100 #200
version = 5  #3
samps = [0, 10,20,50 ] #,100,200,500,1000,2000,5000,10000]
subs = [1,2,3,4,5]
loop = 3 #5
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

# Implement functions that apply a `DetuningLayer` to a matrix containing scaled detunings 
# for each image
function apply_layer(layer::DetuningLayer, x::Matrix{<:Real}, nshots=0)
    iter = SHOW_PROGRESS_BAR ? ProgressBar(1:size(x, 2)) : 1:size(x, 2)
    outs = [apply_layer_sampling(layer, x[:, i][:], nshots) for i in iter]
    return hcat(outs...)
end

# Iterate over each single image
# For more details on simulation, please refer to Bloqade.jl and Yao.jl
function apply_layer_sampling(layer::DetuningLayer, x::Vector{<:Real}, nshots=0)
    # Define Hamiltonian, detunings parameterized in terms of PCA values (x)
    h = rydberg_h(layer.atoms; Δ = x, Ω = layer.Ω) 
    
    # System starts in zero state
    reg = layer.reg
    set_zero_state!(reg) 
    
    t_start = layer.t_start
    t_end = layer.t_end
    t_step = layer.step
    
    # Initialize output vector
    steps = floor(Int, (t_end - t_start) / t_step)
    out = zeros(steps * length(layer.readouts))
    
    # Simulate the quantum evolution with Krylov methods and store the readouts
    i = 1
    prob = KrylovEvolution(reg, layer.t_start:layer.step:layer.t_end, h)
    for (step, reg, _) in prob # Step through the state at each time step 
        step == 1 && continue # ignore first time step, this is just the initial state
        for op in layer.readouts
            if nshots > 0
                ex = sum(measure(op, reg; nshots)) / nshots
                out[i]=real(ex)
            else
                out[i] = real(expect(op, reg)) # store the expectation of each operator for the given state in the output vector
            end 
            i+=1
        end
    end
    return out # Return both the readouts and the final quantum state (wave function)
end

#for loop in 1:5
for subnum in subs
    for samp in samps
        train_X_file = "./X_train_$(recs)rec_sub$(subnum)act$(actfile)v$(version).csv"
        test_X_file = "./X_test_$(recs)rec_sub$(subnum)act$(actfile)v$(version).csv"
        train_y_file = "./y_train_$(recs)rec_sub$(subnum)act$(actfile)v$(version).csv"
        test_y_file = "./y_test_$(recs)rec_sub$(subnum)act$(actfile)v$(version).csv"
        
        # Load data into DataFrames
        df_xlsx_train_X = DataFrame(CSV.File(train_X_file))
        df_xlsx_test_X = DataFrame(CSV.File(test_X_file))
        df_xlsx_train_y = DataFrame(CSV.File(train_y_file))
        df_xlsx_test_y = DataFrame(CSV.File(test_y_file))
        
        names(df_xlsx_test_X)
        
        train_data = DataFrames.select(df_xlsx_train_X, Not(:Column1, :cluspred))
        test_data = DataFrames.select(df_xlsx_test_X, Not(:Column1, :cluspred))
        train_outcomes = DataFrames.select(df_xlsx_train_y, Not(:Column1))
        test_outcomes = DataFrames.select(df_xlsx_test_y, Not(:Column1))
        
        # Inverting matrix and converting all to Float64
        train_inv = Matrix(train_data)'
        test_inv = Matrix(test_data)'
        
        pca_model = fit(PCA, train_inv; maxoutdim=10)
        
        # Transform both train and test data using the PCA model
        train_data_pca = MultivariateStats.transform(pca_model, train_inv)
        test_data_pca = MultivariateStats.transform(pca_model, test_inv)
        
        # Convert back to DataFrames if needed
        train_data_pca_df = DataFrame(train_data_pca, :auto)
        test_data_pca_df = DataFrame(test_data_pca, :auto)
        
        # Print the sizes to verify
        println(size(train_data_pca_df))
        println(size(test_data_pca_df))
        
        Δ_max = 6.0
        
        max_val = maximum(map(col -> maximum(abs, col), eachcol(train_data_pca_df)))
        min_val = maximum(map(col -> minimum(abs, col), eachcol(train_data_pca_df)))
        
        # Compute the spectral value
        spectral = max(max_val, min_val)
        
        # Scale the values to be between [-6.0, 6.0]
        xs = train_data_pca_df ./ spectral .* Δ_max
        xs_matrix = Matrix(xs)
        
        # Define atom positions
        atoms = generate_sites(ChainLattice(), 10; scale = 10) # 10 atoms in a chain
        
        # Create all single site Zᵢ and correlator ZᵢZⱼ readouts 
        nsites = length(atoms)
        readouts = AbstractBlock[put(nsites, i => Z) for i in 1:nsites]
        for i in 1:nsites
            for j in i+1:nsites
                push!(readouts, chain(put(nsites, i => Z), put(nsites, j => Z)))
            end
        end
        
        # Build preprocessing layer 
        pre_layer = DetuningLayer(
            atoms = atoms, 
            readouts = readouts, 
            Ω = 2π, 
            t_start = 0.0, 
            t_end = 4.3, # best 4.3, default 4.0
            step = 0.4, # best 0.4, default 0.5
            reg = zero_state(nsites)
        )
        
        #change this to get results with different number of samples
        NUM_SAMPLES=samp
        embeddings = apply_layer(pre_layer, xs_matrix, NUM_SAMPLES)
        writedlm( "records$(recs)/merck_train_embeddings_recs$(recs)_sub$(subnum)act$(actfile)v$(version)noise_qubits$(dim_input)_milan_sampling_v$(version)_numsamp$(NUM_SAMPLES)_loop$(loop).csv",  embeddings, ',')
        Δ_max = 6.0
        
        #xs = train_inv/spectral * Δ_max # to make sure values to be between [-6.0, 6.0]
        tst_max_val = maximum(map(col -> maximum(abs, col), eachcol(test_data_pca_df)))
        tst_min_val = maximum(map(col -> minimum(abs, col), eachcol(test_data_pca_df)))
        
        # Compute the spectral value
        spectral_tst = max(tst_max_val, tst_min_val)
        
        # Scale the values to be between [-6.0, 6.0]
        xs_tst = test_data_pca_df ./ spectral_tst .* Δ_max
        test_features_qrc = Matrix(xs_tst)
        
        # quantum embeddings for 100 test samples
        test_embeddings = apply_layer(pre_layer, test_features_qrc, NUM_SAMPLES)  
        
        
        writedlm( "records$(recs)/merck_test_embeddings_recs$(recs)_sub$(subnum)act$(actfile)v$(version)noise_qubits$(dim_input)_milan_sampling_v$(version)_numsamp$(NUM_SAMPLES)_loop$(loop).csv",  test_embeddings, ',')
        writedlm("records$(recs)/merck_train_outcomes_recs$(recs)_sub$(subnum)act$(actfile)v$(version)noisequbits$(dim_input)_milan_sampling_v$(version)_numsamp$(NUM_SAMPLES)_loop$(loop).csv", Matrix(train_outcomes), ',')
        writedlm("records$(recs)/merck_test_outcomes_recs$(recs)_sub$(subnum)act$(actfile)v$(version)noisequbits$(dim_input)_milan_sampling_v$(version)_numsamp$(NUM_SAMPLES)_loop$(loop).csv", Matrix(test_outcomes), ',')
    end
end

println("Finished Running")
