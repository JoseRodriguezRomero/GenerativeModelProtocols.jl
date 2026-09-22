using GenerativeModelProtocols
using StatsBase
using FileIO, HDF5
using Lux, Reactant
using DifferentialEquations

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
    protocol = make_empty_data_prot(model, input_size)
    @test_throws Exception train!(protocol; kwargs...)
end

function _base_test_train_model(model, train_data, device; kwargs...)
    protocol = GenerativeModelProtocol(model, train_data; epochs = 20, device = device)
    train!(protocol; kwargs...)

    @test !isempty(protocol._log)
end

function test_train_model(model, train_data; kwargs...)
    println("Testing train! with the default cpu_device()")
    _base_test_train_model(model, train_data, cpu_device(); kwargs...)
end

function test_train_model_reactant(model, train_data; kwargs...)
    println("Testing train! with reactant_device()")
    _base_test_train_model(model, train_data, reactant_device(); kwargs...)
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

# DiffusionModel Utils
function test_compare_tabular_denoiser(denoiser_a::GenerativeModelProtocols.TabularDenoiser, denoiser_b::GenerativeModelProtocols.TabularDenoiser)
    ϵ = 1.0E-9

    @test compare_chains(denoiser_a._ps[].time_embedding_mlp, denoiser_b._ps[].time_embedding_mlp)
    @test compare_layers(denoiser_a._ps[].input_projection, denoiser_b._ps[].input_projection)
    @test compare_tuple_layers(denoiser_a._ps[].residual_layers, denoiser_b._ps[].residual_layers)
    @test compare_tuple_layers(denoiser_a._ps[].time_projection_layers, denoiser_b._ps[].time_projection_layers)
    @test compare_layers(denoiser_a._ps[].output_projection, denoiser_b._ps[].output_projection)
    @test abs(denoiser_a.max_period - denoiser_b.max_period) < ϵ
end

function test_compare_models(model_a::GenerativeModelProtocols.DiffusionModel, model_b::GenerativeModelProtocols.DiffusionModel)
    ϵ = 1.0E-9

    @test maximum(collect(model_a.β) - collect(model_b.β)) < ϵ
    test_compare_tabular_denoiser(model_a.denoiser_model, model_b.denoiser_model)
end

function randomize_tabular_denoiser!(denoiser::GenerativeModelProtocols.TabularDenoiser)
    randomize_chain!(denoiser._ps[].time_embedding_mlp)
    randomize_layer!(denoiser._ps[].input_projection)
    randomize_layers!(denoiser._ps[].residual_layers)
    randomize_layers!(denoiser._ps[].time_projection_layers)
    randomize_layer!(denoiser._ps[].output_projection)
end

function randomize_diffusion_model!(model::GenerativeModelProtocols.DiffusionModel)
    randomize_tabular_denoiser!(model.denoiser_model)
end

# GaussianMixtureModel Utils
function test_compare_models(model_a::GenerativeModelProtocols.GaussianMixtureModel, model_b::GenerativeModelProtocols.GaussianMixtureModel)
    ϵ = 1.0E-8
    
    @test maximum(abs.(model_a.log_σ² - model_b.log_σ²)) < ϵ
    @test maximum(abs.(model_a.μ - model_b.μ)) < ϵ
    @test compare_chains(model_a._ps[].predictor_network, model_b._ps[].predictor_network)
    @test maximum(abs.(model_a.p - model_b.p)) < ϵ
end

function example_chain(input_size::Int, output_size::Int; hidden_layer_size::Int = 16, activation_function::Function = relu)
    return Chain(
        Dense(input_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => output_size),
    )
end

function example_gmm(k::Int, input_size::Int; 
    log_σ²::Union{Matrix{Float32}, Nothing} = nothing, 
    μ::Union{Matrix{Float32}, Nothing} = nothing, 
    predictor_network::Union{Chain, Nothing} = nothing,
    p::Union{Vector{Float32}, Nothing} = nothing)

    if isnothing(log_σ²)
        log_σ² = rand(Float32, k, input_size)
    end

    if isnothing(μ)
        μ = rand(Float32, k, input_size)
    end

    if isnothing(predictor_network)
        predictor_network = example_chain(input_size, k)
    end

    if isnothing(p)
        p = rand(Float32, k)
    end

    return GenerativeModelProtocols.GaussianMixtureModel(;
        log_σ²              = log_σ²,
        μ                   = μ,
        predictor_network   = predictor_network,
        p                   = p
    )
end

# VariationalAutoencoder Utils
function test_compare_models(model_a::GenerativeModelProtocols.VariationalAutoencoder, model_b::GenerativeModelProtocols.VariationalAutoencoder)
    @test compare_tuple_chains(model_a._ps[].encoders, model_b._ps[].encoders)
    @test compare_tuple_chains(model_a._ps[].decoders, model_b._ps[].decoders)
end

function randomize_variational_autoencoder!(model::GenerativeModelProtocols.VariationalAutoencoder)
    randomize_chains!(model._ps[].encoders)
    randomize_chains!(model._ps[].decoders)
end

function example_encoder(num_inputs::Int, latent_dim::Int, latent_layers::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
    first_layer = Chain(
        Dense(num_inputs => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )

    lower_layer = Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )

    lower_layers = [lower_layer for _ in 1:(latent_layers-1)]
    return Tuple(vcat([first_layer], lower_layers))
end

function example_decoder(num_inputs::Int, latent_dim::Int, latent_layers::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
    first_layer = Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => num_inputs)
    )

    lower_layer = Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )

    lower_layers = [lower_layer for _ in 1:(latent_layers-1)]
    return Tuple(vcat([first_layer], lower_layers))
end

# GenerativeAdversarialNetwork Utils
function test_compare_models(model_a::GenerativeModelProtocols.GenerativeAdversarialNetwork, model_b::GenerativeModelProtocols.GenerativeAdversarialNetwork)
    @test compare_chains(model_a._ps_discriminator[].discriminator, model_b._ps_discriminator[].discriminator)
    @test compare_tuple_chains(model_a.vae_model._ps[].encoders, model_b.vae_model._ps[].encoders)
    @test compare_tuple_chains(model_a.vae_model._ps[].decoders, model_b.vae_model._ps[].decoders)
end

function randomize_generative_adversarial_network!(model::GenerativeModelProtocols.GenerativeAdversarialNetwork)
    randomize_chain!(model._ps_discriminator[].discriminator)
    randomize_chains!(model.vae_model._ps[].encoders)
    randomize_chains!(model.vae_model._ps[].decoders)
end

function example_encoder(num_inputs::Int, latent_dim::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
    return Chain(
        Dense(num_inputs => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim, activation_function)
    )
end

function example_generator(num_inputs::Int, latent_dim::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => num_inputs, activation_function)
    )
end

function example_vae(num_inputs::Int, latent_dim::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
    encoder = example_encoder(num_inputs, latent_dim;
        hidden_layer_size   = hidden_layer_size,
        activation_function = activation_function
    )

    decoder = example_generator(num_inputs, latent_dim;
        hidden_layer_size   = hidden_layer_size,
        activation_function = activation_function
    )

    return GenerativeModelProtocols.VariationalAutoencoder((encoder,), (decoder,))
end

function example_critic(num_inputs::Int, output_size::Int = 1; hidden_layer_size::Int = 32, activation_function::Function = relu)
    return Chain(
        Dense(num_inputs => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => output_size, activation_function)
    )
end

# NormalizingFlow Utils
function randomize_normalizing_flow!(model::GenerativeModelProtocols.NormalizingFlow)
    randomize_chain!(model._ps[].velocity_field)
end

function test_compare_models(model_a::GenerativeModelProtocols.NormalizingFlow, model_b::GenerativeModelProtocols.NormalizingFlow)
    @test compare_chains(model_a._ps[].velocity_field, model_b._ps[].velocity_field)
end

