using GenerativeModelProtocols
using FileIO, HDF5
using Flux

using Test

function make_test_train_data(num_samples)
    t = (2.0*π) .* rand(Float64,num_samples)

    x_noise = 0.05 .* randn(Float64,num_samples)
    y_noise = 0.05 .* randn(Float64,num_samples)

    x = cos.(1.0.*t) .+ x_noise
    y = sin.(2.0.*t) .+ y_noise
    
    return collect(transpose(hcat(x,y)))
end

function randomize_chains!(chains::Tuple{Vararg{Chain}})
    for i in eachindex(chains)
        for j in eachindex(chains[i])
            chains[i][j].bias .= rand(Float64, size(chains[i][j].bias))
            chains[i][j].weight .= rand(Float64, size(chains[i][j].weight))
        end
    end
end

function check_file_size(file_path::String)
    return isfile(file_path) && filesize(file_path) > 0
end

function remove_file(file_path::String)
    if isfile(file_path)
        rm(file_path)
    end
end

function compare_chains(chains_a::Tuple{Vararg{Chain}}, chains_b::Tuple{Vararg{Chain}})
    if length(chains_a) != length(chains_b)
        return false
    end

    ϵ = 1.0E-9
    for i in eachindex(chains_a)
        chain_a = chains_a[i]
        chain_b = chains_b[i]

        if length(chain_a) != length(chain_b)
            return false
        end

        for j in eachindex(chain_a)
            layer_a = chain_a[j]
            layer_b = chain_b[j]

            bias_diff = abs(maximum(layer_a.bias - layer_b.bias))
            weight_diff = abs(maximum(layer_a.weight - layer_b.weight))

            if (bias_diff > ϵ) || (weight_diff > ϵ)
                return false
            end
        end
    end

    return true
end

include("gaussian_mixture_model_tests.jl")
include("variational_autoencoder_tests.jl")

