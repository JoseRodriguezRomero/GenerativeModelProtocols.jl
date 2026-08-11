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

function _vae_default_encoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)::Tuple{Vararg{Chain}}
    encoders = Vector{Chain}(undef, latent_layers)
    encoders[1] = _vae_default_encoder_final_network(num_inputs, latent_dim, hidden_layer_size, activation_function)

    if latent_layers > 1
        encoders[2:end] = [_vae_default_encoder_mid_network(latent_dim, hidden_layer_size, activation_function) for _ in 1:(latent_layers-1)]
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

function _vae_default_decoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)::Tuple{Vararg{Chain}}
    decoders = Vector{Chain}(undef, latent_layers)
    decoders[1] = _vae_default_decoder_final_network(num_inputs, latent_dim, hidden_layer_size, activation_function)

    if latent_layers > 1
        decoders[2:end] = [_vae_default_decoder_mid_network(latent_dim, hidden_layer_size, activation_function) for _ in 1:(latent_layers-1)]
    end

    return Tuple(decoders)
end

function compatible_vae_model(encoders::Tuple{Vararg{Chain}}, decoders::Tuple{Vararg{Chain}})::Bool
    if _input_size(encoders[1]) != _output_size(decoders[1])
        return false
    end

    if 2 * _input_size(decoders[end]) != _output_size(encoders[end])
        return false
    end

    if length(encoders) != length(decoders)
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

@compat public VariationalAutoencoder

"""
$TYPEDEF

A structure containing the general parameters needed to evaluate and train 
a Variational Autoencoder (VAE). Once trained, its decoder can be used as a 
generative model.

$TYPEDFIELDS
"""
@kwdef struct VariationalAutoencoder <: AbstractGenerativeModel
    """Encoder chain of the VAE, responsible for encoding time-series data into a latent representation."""
    encoders::Tuple{Vararg{Chain}} 
    """Decoder chain of the VAE, responsible for generating time-series data from the latent representation."""
    decoders::Tuple{Vararg{Chain}}

    function VariationalAutoencoder(encoders::Tuple{Vararg{Chain}}, decoders::Tuple{Vararg{Chain}})
        if !compatible_vae_model(encoders, decoders)
            @error "Incompatible VariationalAutoencoder architecture!"
            throw(MethodError(VariationalAutoencoder, (encoders, decoders)))
        end

        return new(encoders, decoders)
    end
end

"""
    GenerativeModelProtocols.VariationalAutoencoder(input_dim::Int, latent_dim::Int = 1, latent_layers::Int = 1; β::Union{Float64,Vector{Float64}} = 1.0)

Convenience constructor that generates a 
`GenerativeModelProtocols.VariationalAutoencoder` using default encoder and 
decoder network architectures.
"""
function VariationalAutoencoder(input_dim::Int, latent_dim::Int = 1, latent_layers::Int = 1)
    return VariationalAutoencoder(
        _vae_default_encoder_network(input_dim, latent_dim, latent_layers),
        _vae_default_decoder_network(input_dim, latent_dim, latent_layers)
    )
end

function Base.display(model::VariationalAutoencoder)
    print_padding = @_default_print_padding
    println("$(summary(model)):")

    println("encoders: ")
    _print_chains(model.encoders, print_padding)
    println("")

    println("decoders: ")
    _print_chains(model.decoders, print_padding)
end

function _input_size(model::VariationalAutoencoder)::Int
    return _input_size(model.encoders[1])
end

function _latent_size(model::VariationalAutoencoder)::Int
    return _input_size(model.decoders[end])
end 

function load_variational_autoencoder_parameters end

macro load_variational_autoencoder_parameters(saved_model, main_group_name, generative_model_group_name)
    return :(load_variational_autoencoder_parameters($(esc(saved_model)); 
        $(main_group_name = esc(main_group_name)), 
        $(generative_model_group_name = esc(generative_model_group_name))
    ))
end

function VariationalAutoencoder(saved_model::String; 
    main_group_name::String = @default_main_group_name,
    generative_model_group_name::String = @default_generative_model_group_name)

    return @load_variational_autoencoder_parameters(saved_model, main_group_name, generative_model_group_name)
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
    σ_buf[:, 1, :] = exp.(logσ²_buf[:, 1, :] .*  0.5)
    z_buf[:, 1, :] = sample_latent(μ_buf[:, 1, :], σ_buf[:, 1, :])

    for i in 2:num_latent_layers
        enc_out = encoders[i](z_buf[:, i-1, :])
        μ_buf[:, i, :] = enc_out[1:latent_dim, :]
        logσ²_buf[:, i, :] = enc_out[(latent_dim+1):end, :]
        σ_buf[:, i, :] = exp.(logσ²_buf[:, i, :] .*  0.5)
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
        σ_buf[:, idx, :] = exp.(logσ²_buf[:, idx, :] .*  0.5)
    end

    x̂ = decoders[1](z[:, 1, :])
    return copy(μ_buf), copy(σ_buf), copy(logσ²_buf), x̂
end

function _vae_elbo(model::VariationalAutoencoder, β::Float64, x)
    latent_dim = _latent_size(model)
    num_latent_layers = length(model.decoders)

    μ_enc, σ_enc, logσ²_enc, z = _encode_vae(model.encoders, x, latent_dim, num_latent_layers)
    μ_dec, σ_dec, logσ²_dec, x̂ = _decode_vae(model.decoders, z, latent_dim, num_latent_layers)

    recon_loss =  0.5 .* mean(sum((x .- x̂) .^ 2, dims = 1))
    kl_loss =  0.5 .* mean(sum(logσ²_dec .- logσ²_enc .+ (σ_enc.^2 .+ (μ_enc .- μ_dec).^2) ./ σ_dec.^2 .- 1.0, dims = 1))

    return recon_loss + β * kl_loss
end

function load_model!(dst::VariationalAutoencoder, src::VariationalAutoencoder)
    load_model!(dst.encoders, src.encoders)
    load_model!(dst.decoders, src.decoders)
end

function _train!(protocol::GenerativeModelProtocol, model::VariationalAutoencoder, β::Float64; print_log::Bool = true)
    model_train_device = model |> protocol.device
    opt_state = Flux.setup(protocol.optimiser, model_train_device)
    batchsize_device = protocol.batchsize |> protocol.device
    training_data_device = Float64.(protocol.training_data) |> protocol.device
    shuffle_device = protocol.shuffle |> protocol.device

    loader = load_data(training_data_device, batchsize_device, shuffle_device)

    if print_log; println("Training VAE... (β = $β)") end
    for epoch in 1:protocol.epochs
        epoch_loss = 0.0
        total_grad_norm = 0.0

        if epoch % 100 == 0 && protocol.shuffle
            loader = load_data(training_data_device, batchsize_device, shuffle_device)
        end

        for x_batch in loader
            loss, grads = Flux.withgradient(model_train_device) do m
                _vae_elbo(m, β, x_batch)
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
                @printf("Epoch %8d | Avg. ELBO: %16.8e | Grad Norm: %16.8e \n", epoch, average_loss, average_grad_norm)
            end
        end
    end
    if print_log; println("Training complete!") end

    load_model!(model, model_train_device)

    return protocol._log
end

function _train!(protocol::GenerativeModelProtocol, model::VariationalAutoencoder; β::Union{Vector{Float64}, Float64} = 1.0, print_log::Bool = true)
    training_log = nothing
    for i in eachindex(β)
        training_log = _train!(protocol, model, β[i]; print_log = print_log)
    end

    return training_log
end

function _encode(model::VariationalAutoencoder, x::Matrix)
    num_latent_layers = length(model.encoders)
    latent_size = _latent_size(model)

    enc_out = model.encoders[1](x)
    z = sample_latent(enc_out[1:latent_size,:], exp.(enc_out[(latent_size + 1):end,:] .*  0.5))
    
    for i in 2:num_latent_layers
        enc_out = model.encoders[i](z)
        z = sample_latent(enc_out[1:latent_size,:], exp.(enc_out[(latent_size + 1):end,:] .*  0.5))
    end

    return z
end

function _encode(model::VariationalAutoencoder, x::Vector)
    return _encode(model, reshape(x, :, 1))[:]
end

function _decode(model::VariationalAutoencoder, z::Matrix)
    num_latent_layers = length(model.decoders)
    latent_size = _latent_size(model)

    for i in 1:(num_latent_layers-1)
        dec_out = model.decoders[num_latent_layers-i+1](z)
        z = sample_latent(
            dec_out[1:latent_size, :], 
            exp.(dec_out[(latent_size + 1):end, :] .*  0.5)
        )
    end

    return model.decoders[1](z)
end

function _decode(model::VariationalAutoencoder, z::Vector)
    return _decode(model, reshape(z,:,1))[:]
end

function _eval(model::VariationalAutoencoder, n_samples::Int)
    z = randn(Float64, _latent_size(model), n_samples)
    return _decode(model,z)
end

function _eval(model::VariationalAutoencoder)
    return _eval(model,1)[:]
end

