using GenerativeModelProtocols
using StatsBase
using FileIO, HDF5
using Lux

using Test

function make_test_train_data(num_samples)
    t = (2.0*π) .* rand(Float32,num_samples)

    x_noise = 0.05 .* randn(Float32,num_samples)
    y_noise = 0.05 .* randn(Float32,num_samples)

    x = cos.(1.0.*t) .+ x_noise
    y = sin.(2.0.*t) .+ y_noise
    
    return collect(transpose(hcat(x,y)))
end

function randomize_layers!(layers::NamedTuple)
    for i in eachindex(layers)
        layers[i].bias .= rand(Float32, size(layers[i].bias))
        layers[i].weight .= rand(Float32, size(layers[i].weight))
    end
end

function randomize_layer!(layer::NamedTuple)
    randomize_layers!((layer1 = layer,))
end

function randomize_chains!(chains::NamedTuple)
    for i in eachindex(chains)
        randomize_layers!(chains[i])
    end
end

function randomize_chain!(chain::NamedTuple)
    randomize_chains!((chain1 = chain,))
end

function check_file_size(file_path::String)
    return isfile(file_path) && filesize(file_path) > 0
end

function remove_file(file_path::String)
    if isfile(file_path)
        rm(file_path)
    end
end

function compare_layers(layer_a::NamedTuple, layer_b::NamedTuple)
    ϵ = 1.0E-9

    bias_diff = maximum(abs.(layer_a.bias - layer_b.bias))
    weight_diff = maximum(abs.(layer_a.weight - layer_b.weight))

    if (bias_diff > ϵ) || (weight_diff > ϵ)
        return false
    end

    return true
end

function compare_tuple_layers(layers_a::NamedTuple, layers_b::NamedTuple)
    if length(layers_a) != length(layers_b)
        return false
    end

    for i in eachindex(layers_a)
        layer_a = layers_a[i]
        layer_b = layers_b[i]

        if !compare_layers(layer_a, layer_b)
            return false
        end
    end

    return true
end

function compare_chains(chain_a::NamedTuple, chain_b::NamedTuple)
    return compare_tuple_layers(chain_a, chain_b)
end

function compare_tuple_chains(chains_a::NamedTuple, chains_b::NamedTuple)
    if length(chains_a) != length(chains_b)
        return false
    end

    for i in eachindex(chains_a)
        chain_a = chains_a[i]
        chain_b = chains_b[i]

        if length(chain_a) != length(chain_b)
            return false
        end

        for j in eachindex(chain_a)
            layer_a = chain_a[j]
            layer_b = chain_b[j]

            if !compare_layers(layer_a, layer_b)
                return false
            end
        end
    end

    return true
end

function make_empty_data_prot(model::GenerativeModelProtocols.AbstractGenerativeModel, input_size::Int)
    return GenerativeModelProtocol(; 
        model              = model,
        mean_training_data = Tuple(zeros(Float32, input_size)), 
        var_training_data  = Tuple(ones(Float32, input_size))
    )
end

function test_train_model_no_data(model::GenerativeModelProtocols.AbstractGenerativeModel, input_size::Int; kwargs...)
    try
        train!(make_empty_data_prot(model, input_size); kwargs...)
        return true
    catch
        return false
    end
end

function test_train_model(model::GenerativeModelProtocols.AbstractGenerativeModel, train_data::Matrix; kwargs...)
    protocol = GenerativeModelProtocol(model, train_data; epochs = 20)
    train_log = train!(protocol; kwargs...)
    @test isa(train_log, typeof(protocol._log))
end

function test_model_make_synthetic_data(model::GenerativeModelProtocols.AbstractGenerativeModel, input_size::Int)
    protocol = make_empty_data_prot(model, input_size)

    x_synthetic = protocol(100)
    @test isa(x_synthetic, Matrix)
end

function test_model_make_categorical_synthetic_data(model::GenerativeModelProtocols.AbstractGenerativeModel, input_size::Int)
    protocol = make_empty_data_prot(model, input_size)

    for i in 1:5
        if i > latent_size(protocol)
            break
        end

        @test isa(protocol(i, 100), Matrix)
    end
end

function test_model_categorize(model::GenerativeModelProtocols.AbstractGenerativeModel, input_size::Int)
    protocol = make_empty_data_prot(model, input_size)

    for i in 1:5
        if i > latent_size(protocol)
            break
        end

        @test isa(categorize(protocol, protocol(i, 100)), Matrix)
        @test isa(categorize(protocol, protocol()), Vector)
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

function test_display(obj::Any)
    @test isa(capture_display(obj), String)
end

function test_model_display_no_data(model::GenerativeModelProtocols.AbstractGenerativeModel, input_size::Int)
    test_display(make_empty_data_prot(model, input_size))
    test_display(model)
end

function test_model_display(model::GenerativeModelProtocols.AbstractGenerativeModel, input_size::Int)
    train_data = rand(Float32, input_size, 100)
    protocol = GenerativeModelProtocol(;
        model              = model, 
        mean_training_data = Tuple(mean(train_data, dims = 2)), 
        var_training_data  = Tuple(var(train_data, dims = 2))
    )

    test_display(protocol)
    test_display(model)
end

function test_model_save(save_path::String, model::GenerativeModelProtocols.AbstractGenerativeModel, input_size::Int)
    metadata = Dict(
        "A" => "a",
        "B" => 1.0,
        "C" => [1.0, 2.0]
    )

    save(save_path, make_empty_data_prot(model, input_size); metadata = metadata)
    @test check_file_size(save_path)
end

include("diffusion_model_tests.jl")
include("gaussian_mixture_model_tests.jl")
include("variational_autoencoder_tests.jl")
include("generative_adversarial_network_tests.jl")

