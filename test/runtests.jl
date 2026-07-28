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

function randomize_chains!(chain::Chain)
    randomize_chains!((chain,))
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

            bias_diff = maximum(abs.(layer_a.bias - layer_b.bias))
            weight_diff = maximum(abs.(layer_a.weight - layer_b.weight))

            if (bias_diff > ϵ) || (weight_diff > ϵ)
                return false
            end
        end
    end

    return true
end

function compare_chains(chain_a::Chain, chain_b::Chain)
    return compare_chains((chain_a,),(chain_b,))
end

function test_train_model(model::GenerativeModelProtocols.AbstractGenerativeModel, train_data::Matrix{Float64})
    protocol = GenerativeModelProtocol(model, train_data;
        batchsize   = 32,
        epochs      = 20,
        optimiser   = Adam(; eta = 1.0E-3, beta = (0.95, 0.999)),
        device      = cpu_device()
    )

    train_log = train!(protocol)
    @test isa(train_log, typeof(protocol._log))
end

function test_model_make_synthetic_data(model::GenerativeModelProtocols.AbstractGenerativeModel)
    protocol = GenerativeModelProtocol(model)

    x_synthetic = protocol(100)
    @test isa(x_synthetic, Matrix)
end

function test_model_make_categorical_synthetic_data(model::GenerativeModelProtocols.AbstractGenerativeModel)
    protocol = GenerativeModelProtocol(model)

    for i in 1:5
        if i > model.k
            break
        end

        x_synthetic = protocol(i, 100)
        @test isa(x_synthetic, Matrix)
    end
end

function test_model_categorize(model::GenerativeModelProtocols.AbstractGenerativeModel)
    protocol = GenerativeModelProtocol(model)

    for i in 1:5
        if i > model.k
            break
        end

        p_samples = categorize(protocol, protocol(i, 100))
        @test isa(p_samples, Matrix)
    end
end

function capture_display(obj::Any)
    pipe = Pipe()

    redirect_stdout(pipe) do
        display(obj)
    end

    close(pipe.in)
    printed_string = read(pipe.out, String)
    println(printed_string)

    return printed_string
end

function test_model_display(model::GenerativeModelProtocols.AbstractGenerativeModel)
    protocol = GenerativeModelProtocol(model) 

    @test isa(capture_display(protocol), String)
    @test isa(capture_display(model), String)
end

include("gaussian_mixture_model_tests.jl")
include("variational_autoencoder_tests.jl")

