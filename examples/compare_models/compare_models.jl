using GenerativeModelProtocols
using DifferentialEquations
using FileIO, HDF5

using Distances
using OptimalTransport
using Tulip

using Plots
using StatsBase

using Printf

include("make_data.jl")

base_dir = @__DIR__

function make_test_data(num_samples)
    x_test, y_test = make_data(num_samples)
    return collect(transpose(hcat(x_test,y_test)))
end

function compute_distances_base(protocol, dist_func::Function, file_name::String, num_reps::Int, min_num_samples::Int, max_num_samples::Int)
    function compute_dist(num_samples)
        return dist_func(protocol(num_samples), make_test_data(num_samples))
    end

    function compute_dist_reps(num_samples, num_reps)
        return [compute_dist(num_samples) for _ in 1:num_reps]
    end

    print_header(io) = println(io, @sprintf "%20s %20s %20s" "Num Samples" "Mean" "Stnd Dev")
    print_data(io, num_samples, μ, σ) = println(io, @sprintf "%20d %20.8E %20.8E" num_samples μ σ)

    println("file: $(base_dir * "/" * file_name)")
    open(base_dir * "/" * file_name, "w") do io
        
        print_header(io)
        print_header(stdout)

        for num_samples in min_num_samples:min_num_samples:max_num_samples
            samples = compute_dist_reps(num_samples, num_reps)
            μ = mean(samples)
            σ = std(samples)

            print_data(io, num_samples, μ, σ)
            print_data(stdout, num_samples, μ, σ)
        end 
    end

    println("done with $file_name")
end

function compute_distances(dist_func::Function, files_dir::String; num_reps::Int = 200, min_num_samples::Int, max_num_samples::Int)
    protocol = make_test_data
    compute_distances_base(protocol, dist_func, files_dir * "ref_model.txt", num_reps, min_num_samples, max_num_samples)

    protocol = GenerativeModelProtocol(base_dir * "/trained_models/nf_model.h5")
    compute_distances_base(protocol, dist_func, files_dir * "nf_model.txt", num_reps, min_num_samples, max_num_samples)

    protocol = GenerativeModelProtocol(base_dir * "/trained_models/dm_model.h5")
    compute_distances_base(protocol, dist_func, files_dir * "dm_model.txt", num_reps, min_num_samples, max_num_samples)

    protocol = GenerativeModelProtocol(base_dir * "/trained_models/gmm_model.h5")
    compute_distances_base(protocol, dist_func, files_dir * "gmm_model.txt", num_reps, min_num_samples, max_num_samples)

    protocol = GenerativeModelProtocol(base_dir * "/trained_models/vae_model.h5")
    compute_distances_base(protocol, dist_func, files_dir * "vae_model.txt", num_reps, min_num_samples, max_num_samples)

    protocol = GenerativeModelProtocol(base_dir * "/trained_models/gan_model.h5")
    compute_distances_base(protocol, dist_func, files_dir * "gan_model.txt", num_reps, min_num_samples, max_num_samples)
end

# compute_distances(GenerativeModelProtocols.earth_mover_distance, "emd_dist_comp/"; min_num_samples = 50, max_num_samples = 200)
# compute_distances(GenerativeModelProtocols.energy_distance, "energy_dist_comp/"; min_num_samples = 200, max_num_samples = 800)
compute_distances(GenerativeModelProtocols.sinkhorn_distance, "sinkhorn_dist_comp/"; min_num_samples = 200, max_num_samples = 800)

