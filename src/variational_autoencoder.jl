macro _vae_default_activation_function()
    return swish
end

function _vae_default_encoder_final_network(num_inputs::Int, latent_dim::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    return Chain(
        Dense(num_inputs => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )
end

function _vae_default_encoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )
end

function _vae_default_encoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
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
    )
end

function _vae_default_decoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )
end

function _vae_default_decoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    decoders = Vector{Chain}(undef, latent_layers)
    decoders[1] = _vae_default_decoder_final_network(num_inputs, latent_dim, hidden_layer_size, activation_function)

    if latent_layers > 1
        decoders[2:end] = [_vae_default_decoder_mid_network(latent_dim, hidden_layer_size, activation_function) for _ in 1:(latent_layers-1)]
    end

    return Tuple(decoders)
end

function compatible_vae_model(
    encoders::NamedTuple{EncNames, NTuple{N, C}}, 
    decoders::NamedTuple{DecNames, NTuple{N, C}}
    ) where {EncNames, DecNames, N, C}

    if _is_empty_state_tree(encoders) || _is_empty_state_tree(decoders)
        return true
    end

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
@kwdef struct VariationalAutoencoder{EncLayerNames, DecLayerNames} <: AbstractGenerativeModel
    """Encoder chain of the VAE, responsible for encoding time-series data into a latent representation."""
    encoders::NamedTuple{EncLayerNames, <:Tuple{Vararg{Chain}}}
    """Decoder chain of the VAE, responsible for generating time-series data from the latent representation."""
    decoders::NamedTuple{DecLayerNames, <:Tuple{Vararg{Chain}}}
    """Trained parameters of the model. Users should not use this directly."""
    _ps::Union{Ref{<:NamedTuple}, Nothing} = nothing
    """Trained state of the model. Users should not use this directly."""
    _st::Union{Ref{<:NamedTuple}, Nothing} = nothing

    function VariationalAutoencoder(
        encoders::NamedTuple{EncLayerNames, <:Tuple{Vararg{Chain}}}, 
        decoders::NamedTuple{DecLayerNames, <:Tuple{Vararg{Chain}}},
        _ps::Union{Ref{<:NamedTuple}, Nothing},
        _st::Union{Ref{<:NamedTuple}, Nothing}
        ) where {EncLayerNames, DecLayerNames}
        if !compatible_vae_model(encoders, decoders)
            @error "Incompatible VariationalAutoencoder architecture!"
            throw(MethodError(VariationalAutoencoder, (encoders, decoders)))
        end

        if isnothing(_ps) && isnothing(_st)
            _ps_val, _st_val = Lux.setup(Random.default_rng(), (encoders = encoders, decoders = decoders))
            _ps = Ref{NamedTuple}(_ps_val)
            _st = Ref{NamedTuple}(_st_val)
        end

        return new{EncLayerNames,DecLayerNames}(encoders, decoders, _ps, _st)
    end
end

"""
    GenerativeModelProtocols.VariationalAutoencoder(encoders::Tuple{Vararg{Chain}}, decoders::Tuple{Vararg{Chain}})

Convenience constructor that generates a 
`GenerativeModelProtocols.VariationalAutoencoder` from plain `Tuple` containers.
"""
function VariationalAutoencoder(encoders::Tuple{Vararg{Chain}}, decoders::Tuple{Vararg{Chain}})
    return VariationalAutoencoder(;
        encoders = _named_tuples_from_tuple(encoders, "encoder"), 
        decoders = _named_tuples_from_tuple(decoders, "decoder")
    )
end

"""
    GenerativeModelProtocols.VariationalAutoencoder(encoders::NamedTuple{EncLayerNames, Tuple{Vararg{Chain}}}, decoders::NamedTuple{DecLayerNames, Tuple{Vararg{Chain}}}) where {EncLayerNames, DecLayerNames}

Convenience constructor that generates a 
`GenerativeModelProtocols.VariationalAutoencoder` from plain `Tuple` containers.
"""
function VariationalAutoencoder(
    encoders::NamedTuple{EncLayerNames, <:Tuple{Vararg{Chain}}}, 
    decoders::NamedTuple{DecLayerNames, <:Tuple{Vararg{Chain}}}) where {EncLayerNames, DecLayerNames}

    return VariationalAutoencoder(Tuple(encoders), Tuple(decoders))
end

"""
    GenerativeModelProtocols.VariationalAutoencoder(input_dim::Int, latent_dim::Int = 1, latent_layers::Int = 1)

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
    println("$(Base.typename(typeof(model)).wrapper):")

    println("encoders: ")
    _print_chains(model.encoders)
    println("")

    println("decoders: ")
    _print_chains(model.decoders)
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
    ϵ = randn_like(μ, size(μ))
    return μ .+ σ .* ϵ
end

function _encode_vae(encoders::NamedTuple{LayerNames, <:Tuple{Vararg{Chain}}}, x, latent_dim::Int, num_latent_layers::Int, batch_size::Int, ps, st) where {LayerNames}
    T = eltype(x)
    
    μ = similar(x, T, latent_dim, num_latent_layers, batch_size)
    σ = similar(x, T, latent_dim, num_latent_layers, batch_size)
    logσ² = similar(x, T, latent_dim, num_latent_layers, batch_size)
    z = similar(x, T, latent_dim, num_latent_layers, batch_size)

    st_encoders_list = Any[st...]

    k1 = LayerNames[1]
    enc_out, st_new = encoders[k1](x, ps[k1], st[k1])
    st_encoders_list[1] = st_new 

    μ[:, 1, :] .= enc_out[1:latent_dim, :]
    logσ²[:, 1, :] .= enc_out[(latent_dim+1):end, :]
    σ[:, 1, :] .= exp.(logσ²[:, 1, :] .* T(0.5f0))
    z[:, 1, :] .= sample_latent(μ[:, 1, :], σ[:, 1, :])

    for i in 2:num_latent_layers
        ki = LayerNames[i]
        enc_out, st_new = encoders[ki](z[:, i-1, :], ps[ki], st[ki])
        st_encoders_list[i] = st_new 
        
        μ[:, i, :] .= enc_out[1:latent_dim, :]
        logσ²[:, i, :] .= enc_out[(latent_dim+1):end, :]
        σ[:, i, :] .= exp.(logσ²[:, i, :] .* T(0.5f0))
        z[:, i, :] .= sample_latent(μ[:, i, :], σ[:, i, :])
    end

    return μ, σ, logσ², z
end

function _decode_vae(decoders::NamedTuple{LayerNames, <:Tuple{Vararg{Chain}}}, z, latent_dim::Int, num_latent_layers::Int, batch_size::Int, ps, st) where {LayerNames}
    T = eltype(z)

    μ = zeros(T, latent_dim, num_latent_layers, batch_size)
    σ = ones(T, latent_dim, num_latent_layers, batch_size)
    logσ² = zeros(T, latent_dim, num_latent_layers, batch_size)

    st_decoders_list = Any[st...]

    for i in 2:num_latent_layers
        idx = num_latent_layers - i + 1
        
        z_slice = z[:, idx+1, :]

        ki = LayerNames[idx + 1]
        dec_out, st_new = decoders[ki](z_slice, ps[ki], st[ki])
        st_decoders_list[idx + 1] = st_new 
        
        μ[:, idx, :] .= dec_out[1:latent_dim, :]
        logσ²[:, idx, :] .= dec_out[(latent_dim+1):end, :]
        σ[:, idx, :] .= exp.(logσ²[:, idx, :] .* T(0.5))
    end

    k1 = LayerNames[1]
    z_slice = z[:, 1, :]
    x̂, st_new = decoders[k1](z_slice, ps[k1], st[k1])
    st_decoders_list[1] = st_new 

    return μ, σ, logσ², x̂
end

function _vae_elbo(
    encoders::NamedTuple, 
    decoders::NamedTuple, 
    β::AbstractFloat, 
    latent_dim::Int, 
    num_latent_layers::Int, 
    batch_size::Int, 
    x, ps, st)

    T = eltype(x)
    μ_enc, σ_enc, logσ²_enc, z = _encode_vae(encoders, x, latent_dim, num_latent_layers, batch_size, ps.encoders, st.encoders)
    μ_dec, σ_dec, logσ²_dec, x̂ = _decode_vae(decoders, z, latent_dim, num_latent_layers, batch_size, ps.decoders, st.decoders)

    recon_loss = T(0.5) * mean(sum((x .- x̂) .^ 2, dims = 1))
    
    kl_elements = logσ²_dec .- logσ²_enc .+ (σ_enc.^2 .+ (μ_enc .- μ_dec).^2) ./ σ_dec.^2 .- T(1.0)
    kl_loss = T(0.5) * mean(sum(kl_elements, dims = 1))

    return recon_loss + β * kl_loss
end

function _train!(protocol::GenerativeModelProtocol, model::VariationalAutoencoder, β::AbstractFloat; print_log::Bool = true)
    T = eltype(protocol.training_data)

    ps = protocol.precision(model._ps[]) |> protocol.device
    st = protocol.precision(model._st[]) |> protocol.device

    encoders = model.encoders
    decoders = model.decoders

    β_device = T(β) |> protocol.device
    latent_dim_device = _latent_size(model) |> protocol.device
    num_latent_layers = length(model.encoders) |> protocol.device
    training_data_device = protocol.training_data |> protocol.device

    loader = load_data(training_data_device, protocol.batchsize, protocol.shuffle)
    
    function _vae_train_step!(x, p_current, s_current, o_current)
        _objective = (p) -> _vae_elbo(encoders, decoders, β_device, latent_dim_device, num_latent_layers, size(x,2), x, p, s_current)

        loss_val = _objective(p_current)
        loss_grads = Enzyme.make_zero(p_current)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(_objective),
            Enzyme.Active,
            Enzyme.Duplicated(p_current, loss_grads)
        )

        o_updated, p_updated = Optimisers.update(o_current, p_current, loss_grads)

        return loss_val, p_updated, o_updated
    end

    opt_state = _initial_step(model, ps, st, protocol.optimiser)
    _train_step!, opt_state = _train_step_device_dispatch(protocol.device, _vae_train_step!, loader, opt_state)

    if print_log; println("Training VAE... (β = $β)") end
    for epoch in 1:protocol.epochs
        epoch_loss = T(0.0)

        if epoch % 100 == 0 && protocol.shuffle
            loader = load_data(training_data_device, protocol.batchsize, protocol.shuffle)
        end

        for x_batch in loader
            elbo, opt_state = _train_step!(x_batch, opt_state)
            epoch_loss += elbo
        end

        protocol._log.loss[epoch] = epoch_loss / length(loader)

        if epoch % 5 == 0 || epoch == 1
            average_loss = protocol._log.loss[epoch]
            if print_log
                @printf("Epoch %8d | Avg. ELBO: %16.8e \n", epoch, average_loss)
            end
        end
    end
    if print_log; println("Training complete!") end

    model._ps[] = opt_state.parameters
    model._st[] = opt_state.states

    return protocol._log
end

function _train!(protocol::GenerativeModelProtocol, model::VariationalAutoencoder; β::Union{Vector{<:AbstractFloat}, AbstractFloat} = 1.0, print_log::Bool = true)
    training_log = nothing
    for i in eachindex(β)
        training_log = _train!(protocol, model, β[i]; print_log = print_log)
    end

    return training_log
end

function _vae_encode(encoders, ps, st, x)
    num_latent_layers = length(encoders)
    latent_size = round(Int, _output_size(encoders[end]) / 2)
    T = eltype(x)

    enc_out, st_new = encoders[1](x, ps.encoders[1], st.encoders[1])
    st_encoders_list = Any[st.encoders...]
    st_encoders_list[1] = st_new
    
    mu = enc_out[1:latent_size, :]
    logvar = enc_out[(latent_size + 1):end, :]
    z = sample_latent(mu, exp.(logvar .* T(0.5)))
    
    for i in 2:num_latent_layers
        enc_out, st_new = encoders[i](z, ps.encoders[i], st.encoders[i])
        st_encoders_list[i] = st_new
        
        mu = enc_out[1:latent_size, :]
        logvar = enc_out[(latent_size + 1):end, :]
        z = sample_latent(mu, exp.(logvar .* T(0.5)))
    end

    return z
end

function _encode(model::VariationalAutoencoder, x::Matrix)
    return _vae_encode(model.encoders, model._ps[], model._st[], x)
end

function _encode(model::VariationalAutoencoder, x::Vector)
    return _encode(model, reshape(x, :, 1))[:]
end

function _vae_decode(decoders, ps, st, z)
    num_latent_layers = length(decoders)
    latent_size =  _input_size(decoders[end])
    T = eltype(z)

    st_decoders_list = Any[st.decoders...]

    for i in 1:(num_latent_layers - 1)
        layer_idx = num_latent_layers - i + 1
        
        dec_out, st_new = decoders[layer_idx](z, ps.decoders[layer_idx], st.decoders[layer_idx])
        st_decoders_list[layer_idx] = st_new
        
        μ = dec_out[1:latent_size, :]
        log_σ² = dec_out[(latent_size + 1):end, :]
        z = sample_latent(μ, exp.(log_σ² .* T(0.5)))
    end

    out, st_new = decoders[1](z, ps.decoders[1], st.decoders[1])
    st_decoders_list[1] = st_new

    return out
end

function _decode(model::VariationalAutoencoder, z::Matrix)
    return _vae_decode(model.decoders, model._ps[], model._st[], z)
end

function _decode(model::VariationalAutoencoder, z::Vector)
    return _decode(model, reshape(z,:,1))[:]
end

function _eval(model::VariationalAutoencoder, n_samples::Int)
    ps = model._ps[]

    z = randn_like(ps.encoders.encoder_1.layer_1.weight, (_latent_size(model), n_samples))
    return _decode(model, z)
end

function _eval(model::VariationalAutoencoder)
    return _eval(model, 1)[:]
end

