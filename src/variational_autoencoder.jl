macro _vae_default_activation_function()
    return relu
end

function _vae_default_encoder_final_network(num_inputs::Int, latent_dim::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    return Chain(
        Dense(num_inputs => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    ) |> f64
end

function _vae_default_encoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function), 
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    ) |> f64
end

function _vae_default_encoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, hidden_layer_size::Int = 32)::Tuple{Vararg{Chain}}
    encoders = Vector{Chain}(undef, latent_layers)
    encoders[1] = _vae_default_encoder_final_network(num_inputs, latent_dim, hidden_layer_size)

    if latent_layers > 1
        encoders[2:end] = [_vae_default_encoder_mid_network(latent_dim, hidden_layer_size) for _ in 1:(latent_layers-1)]
    end

    return Tuple(encoders)
end

function _vae_default_decoder_final_network(num_inputs::Int, latent_dim::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => num_inputs)
    ) |> f64
end

function _vae_default_decoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    ) |> f64
end

function _vae_default_decoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, hidden_layer_size::Int = 32)::Tuple{Vararg{Chain}}
    decoders = Vector{Chain}(undef, latent_layers)
    decoders[1] = _vae_default_decoder_final_network(num_inputs, latent_dim, hidden_layer_size)

    if latent_layers > 1
        decoders[2:end] = [_vae_default_decoder_mid_network(latent_dim, hidden_layer_size) for _ in 1:(latent_layers-1)]
    end

    return Tuple(decoders)
end

function compatible_vae_model(encoders::Tuple{Vararg{Chain}}, decoders::Tuple{Vararg{Chain}})::Bool
    enc_input_dim = size(encoders[1][1].weight)[2]
    enc_latent_dim = round(Int, size(encoders[end][end].weight)[1] / 2.0)
    enc_latent_layers = length(encoders)

    dec_input_dim = size(decoders[1][end].weight)[1]
    dec_latent_dim = size(decoders[end][1].weight)[2]
    dec_latent_layers = length(decoders)

    if dec_input_dim != enc_input_dim
        return false
    end

    if dec_latent_dim != enc_latent_dim
        return false
    end

    if dec_latent_layers != enc_latent_layers
        return false
    end

    if !compatible_neural_networks(encoders)
        return false
    end

    if !compatible_neural_networks(decoders)
        return false
    end

    return true
end

public VariationalAutoencoder

"""
$TYPEDEF

A structure containing the general parameters needed to evaluate and train 
a Variational Autoencoder (VAE). Once trained, its decoder can be used as a 
generative model.

$TYPEDFIELDS
"""
@kwdef struct VariationalAutoencoder <: AbstractGenerativeModel
    """Dimensionality of the latent space representation."""
    latent_dim::Int
    """Number of latent layers in the encoder and decoder chains of the VAE."""
    latent_layers::Int
    """Encoder chain of the VAE, responsible for encoding time-series data into a latent representation."""
    encoders::Tuple{Vararg{Chain}} 
    """Decoder chain of the VAE, responsible for generating time-series data from the latent representation."""
    decoders::Tuple{Vararg{Chain}}
    """Weighting factor for the KL divergence term in the VAE loss function."""
    β::Union{Float64,Vector{Float64}} = 1.0
end

"""
    GenerativeModelProtocols.VariationalAutoencoder(encoders::Tuple{Vararg{Chain}}, decoders::Tuple{Vararg{Chain}}; β::Union{Float64,Vector{Float64}} = 1.0)

Convenience constructor to create a 
`GenerativeModelProtocols.VariationalAutoencoder`.

Builds the VAE using user-defined encoder and decoder architectures. Validates 
structural compatibility between the two networks and raises a `MethodError` if 
they cannot be linked. The user is responsible for ensuring the architectures 
align structurally.
"""
function VariationalAutoencoder(
    encoders::Tuple{Vararg{Chain}},
    decoders::Tuple{Vararg{Chain}};
    β::Union{Float64,Vector{Float64}} = 1.0)

    if !compatible_vae_model(encoders, decoders)
        @error "Incompatible Encoder/Decoder architecture!"
        throw(MethodError(VariationalAutoencoder, (encoders, decoders), (β,)))
    end

    VariationalAutoencoder(
        latent_dim = size(decoders[1][1].weight)[2], 
        latent_layers = length(decoders), 
        encoders = encoders, 
        decoders = decoders, 
        β = β
    )
end

"""
    GenerativeModelProtocols.VariationalAutoencoder(input_dim::Int, latent_dim::Int = 1, latent_layers::Int = 1; β::Union{Float64,Vector{Float64}} = 1.0)

Convenience constructor that generates a 
`GenerativeModelProtocols.VariationalAutoencoder` using default encoder and 
decoder network architectures.
"""
function VariationalAutoencoder(input_dim::Int, latent_dim::Int = 1, latent_layers::Int = 1; β::Union{Float64,Vector{Float64}} = 1.0)
    return VariationalAutoencoder(
        _vae_default_encoder_network(input_dim, latent_dim, latent_layers),
        _vae_default_decoder_network(input_dim, latent_dim, latent_layers);
        β = β
    )
end

function Base.display(model::VariationalAutoencoder)
    print_padding = @_default_print_padding
    println("GenerativeModelProtocols.VariationalAutoencoder:")
    println("β             = $(model.β)")
    println("latent_dim    = $(model.latent_dim)")
    println("latent_layers = $(model.latent_layers)")
    println("")

    println("encoders: ")
    _print_chains(model.encoders, print_padding)
    println("")

    println("decoders: ")
    _print_chains(model.decoders, print_padding)
end

function load_variational_autoencoder_parameters(saved_model::Any;
    main_group_name::String = @default_main_group_name,
    generative_model_group_name::String = @default_generative_model_group_name)
    throw(ArgumentError("Types $(typeof(saved_model)) does not implement the required `load_variational_autoencoder_parameters` interface."))
end

function VariationalAutoencoder(saved_model::Any; β::Union{Float64,Vector{Float64}} = 1.0)
    autoencoder_parameters = load_variational_autoencoder_parameters(saved_model)
    encoders = Vector{Chain}(undef, length(autoencoder_parameters.encoders))
    decoders = Vector{Chain}(undef, length(autoencoder_parameters.decoders))

    for i in eachindex(autoencoder_parameters.encoders)
        encoders[i] = Chain([layer_parameters(layer) for layer in autoencoder_parameters.encoders[i].layers]...)
    end

    for i in eachindex(autoencoder_parameters.decoders)
        decoders[i] = Chain([layer_parameters(layer) for layer in autoencoder_parameters.decoders[i].layers]...)
    end

    return VariationalAutoencoder(;
        latent_dim = size(decoders[1][1].weight)[2], 
        latent_layers = length(decoders),
        encoders = Tuple(encoders), 
        decoders = Tuple(decoders),
        β = β
    )
end

function _generative_model(::VariationalAutoencoder)::GenerativeModel
    return variational_autoencoder
end

function sample_latent(μ, σ)
    ϵ = Flux.randn_like(μ, size(μ))
    return μ .+ σ .* ϵ
end

function _encode_vae(encoders::Tuple{Vararg{Chain}}, x, latent_dim, num_latent_layers)
    batch_size = size(x, 2)
    
    μ_buf = Zygote.Buffer(x, eltype(x), latent_dim, num_latent_layers, batch_size)
    σ_buf = Zygote.Buffer(x, eltype(x), latent_dim, num_latent_layers, batch_size)
    logσ²_buf = Zygote.Buffer(x, eltype(x), latent_dim, num_latent_layers, batch_size)
    z_buf = Zygote.Buffer(x, eltype(x), latent_dim, num_latent_layers, batch_size)

    enc_out = encoders[1](x)
    μ_buf[:, 1, :] = enc_out[1:latent_dim, :]
    logσ²_buf[:, 1, :] = enc_out[(latent_dim+1):end, :]
    σ_buf[:, 1, :] = exp.(logσ²_buf[:, 1, :] .* 0.5f0)
    z_buf[:, 1, :] = sample_latent(μ_buf[:, 1, :], σ_buf[:, 1, :])

    for i in 2:num_latent_layers
        enc_out = encoders[i](z_buf[:, i-1, :])
        μ_buf[:, i, :] = enc_out[1:latent_dim, :]
        logσ²_buf[:, i, :] = enc_out[(latent_dim+1):end, :]
        σ_buf[:, i, :] = exp.(logσ²_buf[:, i, :] .* 0.5f0)
        z_buf[:, i, :] = sample_latent(μ_buf[:, i, :], σ_buf[:, i, :])
    end

    return copy(μ_buf), copy(σ_buf), copy(logσ²_buf), copy(z_buf)
end

function _decode_vae(decoders::Tuple{Vararg{Chain}}, z, latent_dim, num_latent_layers)
    batch_size = size(z, 3)
    
    μ_buf = Zygote.Buffer(z, eltype(z), latent_dim, num_latent_layers, batch_size)
    σ_buf = Zygote.Buffer(z, eltype(z), latent_dim, num_latent_layers, batch_size)
    logσ²_buf = Zygote.Buffer(z, eltype(z), latent_dim, num_latent_layers, batch_size)

    μ_buf[:, num_latent_layers, :] = Flux.zeros_like(z, (latent_dim, batch_size))
    σ_buf[:, num_latent_layers, :] = Flux.ones_like(z, (latent_dim, batch_size))
    logσ²_buf[:, num_latent_layers, :] = Flux.zeros_like(z, (latent_dim, batch_size))

    for i in 2:num_latent_layers
        idx = num_latent_layers - i + 1
        dec_out = decoders[idx+1](z[:, idx+1, :])
        μ_buf[:, idx, :] = dec_out[1:latent_dim, :]
        logσ²_buf[:, idx, :] = dec_out[(latent_dim+1):end, :]
        σ_buf[:, idx, :] = exp.(logσ²_buf[:, idx, :] .* 0.5f0)
    end

    x̂ = decoders[1](z[:, 1, :])
    return copy(μ_buf), copy(σ_buf), copy(logσ²_buf), x̂
end

function vae_loss(model::VariationalAutoencoder, β::Float64, input_weights, x)
    batch_size = size(x, 2)
    latent_dim = size(model.decoders[1][1].weight, 2)
    num_latent_layers = length(model.decoders)

    μ_enc, σ_enc, logσ²_enc, z = _encode_vae(model.encoders, x, latent_dim, num_latent_layers)
    μ_dec, σ_dec, logσ²_dec, x̂ = _decode_vae(model.decoders, z, latent_dim, num_latent_layers)

    recon_loss = 0.5f0 .* sum(input_weights .* ((x .- x̂).^2))   
    kl_loss = 0.5f0 .* sum(logσ²_dec .- logσ²_enc .+ (σ_enc.^2 .+ (μ_enc .- μ_dec).^2) ./ σ_dec.^2 .- 1.0f0)

    return (recon_loss + β * kl_loss) / batch_size
end

function _train!(protocol::GenerativeModelProtocol, model::VariationalAutoencoder, β::Float64; print_log::Bool = true)
    model_train_device = model |> protocol.device
    opt_state = Flux.setup(protocol.optimiser, model_train_device)
    batchsize_device = protocol.batchsize |> protocol.device
    training_data_device = Float64.(protocol.training_data) |> protocol.device
    
    loader = Flux.DataLoader(
        training_data_device, 
        batchsize = batchsize_device, 
        shuffle = false,
        parallel = true
    )

    input_weights = [1.0 / var(protocol.training_data[i,:]) for i in 1:size(protocol.training_data,1)]
    input_weights[input_weights .> 1.0E9] .= 0.0 # ignore nearly deterministic inputs
    input_weights_device = input_weights |> protocol.device

    if print_log; println("Training VAE... (β = $β)") end
    for epoch in 1:protocol.epochs
        epoch_loss = 0f0
        total_grad_norm = 0f0

        if epoch % 100 == 0 && protocol.shuffle
            loader = Flux.DataLoader(
                shuffleobs(training_data_device), 
                batchsize = batchsize_device, 
                shuffle = false,
                parallel = true
            )
        end

        for x_batch in loader
            loss, grads = Flux.withgradient(model_train_device) do m
                vae_loss(m, β, input_weights_device, x_batch)
            end

            raw_gradient_arrays = Optimisers.trainables(grads[1])
            batch_grad_norm = sqrt(sum(sum(abs2, g) for g in raw_gradient_arrays if g isa AbstractArray))

            Flux.update!(opt_state, model_train_device, grads[1])
            epoch_loss += loss
            total_grad_norm += batch_grad_norm
        end

        protocol._log.loss[epoch] = epoch_loss / length(loader)
        protocol._log.loss_grad_norm[epoch] = total_grad_norm / length(loader)

        if epoch % 5 == 0 || epoch == 1
            average_loss = protocol._log.loss[epoch]
            average_grad_norm = protocol._log.loss_grad_norm[epoch]
            if print_log
                @printf("Epoch %8d | Average Loss: %16.8e | Grad Norm: %16.8e \n", epoch, average_loss, average_grad_norm)
            end
        end
    end
    if print_log; println("Training complete!") end

    for i in eachindex(model.encoders)
        Flux.loadmodel!(model.encoders[i], model_train_device.encoders[i])
    end

    for i in eachindex(model.decoders)
        Flux.loadmodel!(model.decoders[i], model_train_device.decoders[i])
    end

    return protocol._log
end

function _train!(protocol::GenerativeModelProtocol, model::VariationalAutoencoder, β::Vector; print_log::Bool = true)
    training_log = nothing
    for i in eachindex(β)
        training_log = _train!(protocol, model, β[i]; print_log = print_log)
    end

    return training_log
end

function _train!(protocol::GenerativeModelProtocol, model::VariationalAutoencoder; print_log::Bool = true)
    return _train!(protocol, model, model.β; print_log = print_log)
end

public decode

"""
    decode(model::GenerativeModelProtocols.VariationalAutoencoder, z)

Decodes the latent space representations `z` back into the data space.
"""
function decode(model::VariationalAutoencoder, z)
    num_latent_layers = length(model.decoders)

    for i in 1:(num_latent_layers-1)
        dec_out = model.decoders[num_latent_layers-i+1](z)
        z = sample_latent(dec_out[1:model.latent_dim], exp.(dec_out[(model.latent_dim+1):end] .* 0.5f0))
    end

    return model.decoders[1](z)
end

public encode

"""
    encode(model::GenerativeModelProtocols.VariationalAutoencoder, x)

Encodes the data space variable `x` into a latent space variable.
"""
function encode(model::VariationalAutoencoder, x)
    num_latent_layers = length(model.encoders)

    enc_out = model.encoders[1](x)
    z = sample_latent(enc_out[1:model.latent_dim,:], exp.(enc_out[(model.latent_dim+1):end,:] .* 0.5f0))
    
    for i in 2:num_latent_layers
        enc_out = model.encoders[i](z)
        z = sample_latent(enc_out[1:model.latent_dim,:], exp.(enc_out[(model.latent_dim+1):end,:] .* 0.5f0))
    end

    return z
end

function _eval(protocol::GenerativeModelProtocol, model::VariationalAutoencoder, n_samples::Int)
    window_size = size(protocol.training_data,1)
    sim_log = zeros(Float64,n_samples,window_size)

    for i in 1:n_samples
        z = randn(model.latent_dim)
        sim_log[i,:] = decode(model,z)
    end

    return sim_log
end

struct VariationalAutoencoderParameters
    encoders::Vector{ChainParameters}
    decoders::Vector{ChainParameters}
end

function variational_autoencoder_parameters(model::VariationalAutoencoder)::VariationalAutoencoderParameters
    return VariationalAutoencoderParameters(
        [chain_parameters(encoder) for encoder in model.encoders],
        [chain_parameters(decoder) for decoder in model.decoders]
    )
end

