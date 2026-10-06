macro _vae_default_activation_function()
    return swish
end

function _vae_default_encoder_final_network(num_inputs::Int, latent_dim::Int, 
    hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    
    return Chain(
        Dense(num_inputs => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )
end

function _vae_default_encoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32, 
    activation_function::Function = @_vae_default_activation_function)
    
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )
end

function _vae_default_encoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, 
    hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    encoders = Vector{Chain}(undef, latent_layers)
    
    encoders[1] = _vae_default_encoder_final_network(
        num_inputs, latent_dim, hidden_layer_size, activation_function
    )

    if latent_layers > 1
        encoders[2:end] = [
            _vae_default_encoder_mid_network(latent_dim, hidden_layer_size, activation_function) 
            for _ in 1:(latent_layers-1)
        ]
    end

    return Tuple(encoders)
end

function _vae_default_decoder_final_network(num_inputs::Int, latent_dim::Int, 
    hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => num_inputs)
    )
end

function _vae_default_decoder_mid_network(latent_dim::Int, hidden_layer_size::Int = 32, 
    activation_function::Function = @_vae_default_activation_function)
    
    return Chain(
        Dense(latent_dim => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2*latent_dim)
    )
end

function _vae_default_decoder_network(num_inputs::Int, latent_dim::Int, latent_layers::Int, 
    hidden_layer_size::Int = 32, activation_function::Function = @_vae_default_activation_function)
    
    decoders = Vector{Chain}(undef, latent_layers)
    decoders[1] = _vae_default_decoder_final_network(
        num_inputs, latent_dim, hidden_layer_size, activation_function
    )

    if latent_layers > 1
        decoders[2:end] = [_vae_default_decoder_mid_network(
            latent_dim, hidden_layer_size, activation_function) 
            for _ in 1:(latent_layers-1)
        ]
    end

    return Tuple(decoders)
end

function compatible_vae_model(encoders::NamedTuple, decoders::NamedTuple)

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

# Structure Fields
$TYPEDFIELDS
"""
@kwdef struct VariationalAutoencoder{EncLayerNames, DecLayerNames} <: AbstractGenerativeModel
    """Encoder chain of the VAE, responsible for encoding time-series data into a latent representation."""
    encoders::NamedTuple{EncLayerNames, <:Tuple{Vararg{Chain}}}
    """Decoder chain of the VAE, responsible for generating time-series data from the latent representation."""
    decoders::NamedTuple{DecLayerNames, <:Tuple{Vararg{Chain}}}
    """Trained parameters of the model. Users should not use this directly."""
    _ps::Ref{<:NamedTuple} = Ref{NamedTuple}(NamedTuple())
    """Trained state of the model. Users should not use this directly."""
    _st::Ref{<:NamedTuple} = Ref{NamedTuple}(NamedTuple())

    function VariationalAutoencoder(
        encoders::NamedTuple{EncLayerNames, <:Tuple{Vararg{Chain}}}, 
        decoders::NamedTuple{DecLayerNames, <:Tuple{Vararg{Chain}}},
        _ps::Ref{<:NamedTuple},
        _st::Ref{<:NamedTuple}
        ) where {EncLayerNames, DecLayerNames}
        if !compatible_vae_model(encoders, decoders)
            @error "Incompatible VariationalAutoencoder architecture!"
            throw(MethodError(VariationalAutoencoder, (encoders, decoders)))
        end

        if isempty(_ps[]) || isempty(_st[])
            _ps_val, _st_val = Lux.setup(Random.default_rng(), (encoders = encoders, decoders = decoders))
            _ps[] = _ps_val
            _st[] = _st_val
        end

        return new{EncLayerNames,DecLayerNames}(encoders, decoders, _ps, _st)
    end
end

VariationalAutoencoder{EncLayerNames, DecLayerNames}(args...; kwargs...) where {EncLayerNames, DecLayerNames} = VariationalAutoencoder(args...; kwargs...)

"""
    GenerativeModelProtocols.VariationalAutoencoder(
        encoders::Tuple{Vararg{Lux.Chain}}, decoders::Tuple{Vararg{Lux.Chain}})

Convenience constructor that generates a 
`GenerativeModelProtocols.VariationalAutoencoder` from plain `Tuple` containers.

# Arguments
 - `encoders::Tuple{Vararg{Lux.Chain}}`: The encoder networks.
 - `decoders::Tuple{Vararg{Lux.Chain}}`: The decoder networks.
"""
function VariationalAutoencoder(encoders::Tuple{Vararg{Chain}}, decoders::Tuple{Vararg{Chain}})
    return VariationalAutoencoder(;
        encoders = _named_tuples_from_tuple(encoders, "encoder"), 
        decoders = _named_tuples_from_tuple(decoders, "decoder")
    )
end

"""
    GenerativeModelProtocols.VariationalAutoencoder(
        encoders::NamedTuple{EncLayerNames, Tuple{Vararg{Lux.Chain}}}, 
        decoders::NamedTuple{DecLayerNames, Tuple{Vararg{Lux.Chain}}}) 
        where {EncLayerNames, DecLayerNames}

Convenience constructor that generates a 
`GenerativeModelProtocols.VariationalAutoencoder` from plain `Tuple` containers.

# Arguments
 - `encoders::NamedTuple{EncLayerNames, Tuple{Vararg{Lux.Chain}}}`: The encoder networks.
 - `decoders::NamedTuple{DecLayerNames, Tuple{Vararg{Lux.Chain}}}`: The decoder networks.
"""
function VariationalAutoencoder(
    encoders::NamedTuple{EncLayerNames, <:Tuple{Vararg{Chain}}}, 
    decoders::NamedTuple{DecLayerNames, <:Tuple{Vararg{Chain}}}) where {EncLayerNames, DecLayerNames}

    return VariationalAutoencoder(Tuple(encoders), Tuple(decoders))
end

"""
    GenerativeModelProtocols.VariationalAutoencoder(
        input_dim::Int, latent_dim::Int = 1, latent_layers::Int = 1)

Convenience constructor that generates a 
`GenerativeModelProtocols.VariationalAutoencoder` using default encoder and 
decoder network architectures.

# Arguments
 - `input_dim::Int`: The dimensionality of the input data.
 - `latent_dim::Int`: The dimensionality of the latent space(s) (default is 1).
 - `latent_layers::Int`: The number of latent layers (default is 1).
"""
function VariationalAutoencoder(input_dim::Int, latent_dim::Int = 1, latent_layers::Int = 1)
    return VariationalAutoencoder(
        _vae_default_encoder_network(input_dim, latent_dim, latent_layers),
        _vae_default_decoder_network(input_dim, latent_dim, latent_layers)
    )
end

function _default_optimiser(::VariationalAutoencoder)
    return Adam(; eta = 1.0E-4, beta = (0.95,0.999))
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

function sample_latent(μ, σ, rng)
    ϵ = similar(μ)
    randn!(rng, ϵ)

    return μ .+ σ .* ϵ, rng
end

function _sample_vae_latent_variables(x, latent_dim::Int, num_latent_layers::Int, batch_size::Int, rng)
    ϵ = ntuple(_ -> begin
        ϵ_n = similar(x, eltype(x), latent_dim, batch_size)
        randn!(rng, ϵ_n)
        ϵ_n
    end, num_latent_layers)

    return ϵ, rng
end

@inline function _make_vae_latent_variables_layers(::Val{I}, ::Val{N}, encoders, x, z, ϵ, p, s, latent_dim, T) where {I, N}

    if I <= N
        encoder_names = keys(encoders)
        encoder_name = encoder_names[I]
        input = I == 1 ? x : z[I - 1]
        enc_out, _ = encoders[encoder_name](
            input, p.encoders[encoder_name], s.encoders[encoder_name]
        )
        μ = enc_out[1:latent_dim, :]
        logσ² = enc_out[(latent_dim + 1):end, :]
        σ = exp.(logσ² .* T(0.5))
        zᵢ = μ .+ σ .* ϵ[I]

        return _make_vae_latent_variables_layers(
            Val(I + 1), Val(N), encoders, x, (z..., zᵢ), ϵ, p, s, latent_dim, T
        )
    end

    return z
end

function _make_vae_latent_variables(encoders, latent_dim::Int, num_latent_layers::Int)
    return function (x, ϵ, p, s)
        T = eltype(x)
        return _make_vae_latent_variables_layers(
            Val(1), Val(num_latent_layers), encoders, x, (), ϵ, p, s, latent_dim, T
        )
    end
end

@inline function _encode_vae_layers(::Val{I}, ::Val{N}, encoders, x, z, latent_dim, T, ps, st, st_encoders_list) where {I, N}
    if I <= N
        input = I == 1 ? x : z[I - 1]
        enc_out, st_new = encoders[I](input, ps[I], st[I])
        st_encoders_list[I] = st_new
        μ = enc_out[1:latent_dim, :]
        logσ² = enc_out[(latent_dim + 1):end, :]
        σ = exp.(logσ² .* T(0.5f0))

        μ_rest, σ_rest, logσ²_rest = _encode_vae_layers(
            Val(I + 1), Val(N), encoders, x, z, latent_dim, T, ps, st,
            st_encoders_list
        )
        return (μ, μ_rest...), (σ, σ_rest...), (logσ², logσ²_rest...)
    end

    return (), (), ()
end

function _encode_vae(encoders::NamedTuple{LayerNames, <:Tuple{Vararg{Chain}}}, latent_dim::Int) where {LayerNames}
    return function (x, z, ps, st)
        T = eltype(x)
        st_encoders_list = Any[st...]

        μ, σ, logσ² = _encode_vae_layers(
            Val(1), Val(length(LayerNames)), encoders, x, z, latent_dim, T, ps,
            st, st_encoders_list
        )

        return μ, σ, logσ², NamedTuple{keys(st)}(st_encoders_list)
    end
end

@inline function _decode_vae_layers(::Val{I}, decoders, z, latent_dim, T, ps, st, st_decoders_list) where {I}
    if I > 1
        dec_out, st_new = decoders[I](z[I], ps[I], st[I])
        st_decoders_list[I] = st_new
        μ = dec_out[1:latent_dim, :]
        logσ² = dec_out[(latent_dim + 1):end, :]
        σ = exp.(logσ² .* T(0.5))

        μ_rest, σ_rest, logσ²_rest = _decode_vae_layers(
            Val(I - 1), decoders, z, latent_dim, T, ps, st,
            st_decoders_list
        )
        return (μ, μ_rest...), (σ, σ_rest...), (logσ², logσ²_rest...)
    end

    return (), (), ()
end

function _decode_vae(decoders::NamedTuple{LayerNames, <:Tuple{Vararg{Chain}}}, latent_dim::Int) where {LayerNames}
    return function (z, ps, st)
        T = eltype(first(z))
        st_decoders_list = Any[st...]

        μ_rev, σ_rev, logσ²_rev = _decode_vae_layers(
            Val(length(LayerNames)), decoders, z, latent_dim, T, ps, st,
            st_decoders_list
        )
        μ = (reverse(μ_rev)..., zero.(z[end]))
        σ = (reverse(σ_rev)..., one.(z[end]))
        logσ² = (reverse(logσ²_rev)..., zero.(z[end]))

        x̂, st_new = decoders[1](z[1], ps[1], st[1])
        st_decoders_list[1] = st_new

        return μ, σ, logσ², x̂, NamedTuple{keys(st)}(st_decoders_list)
    end
end

function _vae_elbo( encode_vae, decode_vae, β::AbstractFloat, x, z, ps, st)
    T = eltype(x)
    μ_enc, σ_enc, logσ²_enc, _st_enc = encode_vae(x, z, ps.encoders, st.encoders)
    μ_dec, σ_dec, logσ²_dec, x̂, _st_dec = decode_vae(z, ps.decoders, st.decoders)

    recon_loss = T(0.5) * mean(sum((x .- x̂) .^ 2, dims = 1))

    kl_sum = zero(T)
    for i in eachindex(μ_enc)
        kl_elements = logσ²_dec[i] .- logσ²_enc[i] .+
            (σ_enc[i].^2 .+ (μ_enc[i] .- μ_dec[i]).^2) ./ σ_dec[i].^2 .- T(1.0)
        kl_sum += sum(kl_elements)
    end
    kl_loss = T(0.5) * kl_sum / (size(x, 2) * length(μ_enc))

    _st = (encoders = _st_enc, decoders = _st_dec)

    return recon_loss + β * kl_loss, _st
end

function _vae_objective(make_latent_variables, encode_vae, decode_vae, β)
    return function (x, ϵ, p, s)
        z = make_latent_variables(x, ϵ, p, s)
        return _vae_elbo(encode_vae, decode_vae, β, x, z, p, s)
    end
end

function _train!(protocol::GenerativeModelProtocol, model::VariationalAutoencoder; 
    β::AbstractFloat = 1.0, print_log::Bool, epochs::Int, batchsize::Int, 
    shuffle::Bool, optimiser::AbstractRule, device::MLDataDevices.AbstractDevice, 
    precision::Function)
    
    T = eltype(precision([1.0]))

    ps = precision(model._ps[]) |> device
    st = precision(model._st[]) |> device

    encoders = model.encoders
    decoders = model.decoders

    β_device = T(β) |> device
    latent_dim = _latent_size(model)
    num_latent_layers = length(model.encoders)

    loader = load_data(T.(protocol.training_data), batchsize, shuffle) |> device
    x_init = first(loader)
    z_init = ntuple(
        _ -> fill!(
            similar(x_init, T, latent_dim, size(x_init, 2)), zero(T)
        ),
        num_latent_layers
    )
    ϵ_init = ntuple(
        _ -> fill!(similar(x_init, T, latent_dim, size(x_init, 2)), zero(T)),
        num_latent_layers
    )

    make_latent_variables = _function_device_dispatch(
        device, _make_vae_latent_variables(encoders, latent_dim, num_latent_layers),
        x_init, ϵ_init, ps, st
    )

    decode_vae = _function_device_dispatch(
        device, _decode_vae(decoders, latent_dim), 
        z_init, ps.decoders, st.decoders
    )

    encode_vae = _function_device_dispatch(
        device, _encode_vae(encoders, latent_dim),
        x_init, z_init, ps.encoders, st.encoders
    )

    vae_objective = _function_device_dispatch(
        device, _vae_objective(
            make_latent_variables, encode_vae, decode_vae, β_device
        ),
        x_init, ϵ_init, ps, st
    )

    protocol._log["Mean ELBO"] = zeros(T, epochs)
    
    function _vae_train_step!(x, p_current, s_current, o_current, rng)
        rng_trace = Lux.replicate(rng)
        ϵ, rng_trace = _sample_vae_latent_variables(
            x, latent_dim, num_latent_layers, size(x, 2), rng_trace
        )

        loss_grads = Enzyme.make_zero(p_current)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const((x, ϵ, p, s) -> vae_objective(x, ϵ, p, s)[1]),
            Enzyme.Active,
            Enzyme.Const(x),
            Enzyme.Const(ϵ),
            Enzyme.Duplicated(p_current, loss_grads),
            Enzyme.Const(s_current),
        )

        loss, s_updated = vae_objective(x, ϵ, p_current, s_current)
        o_updated, p_updated = Optimisers.update(o_current, p_current, loss_grads)

        return loss, p_updated, s_updated, o_updated, rng_trace
    end

    opt_state = _initial_step(model, ps, st, optimiser)
    _train_step!, opt_state, rng = _train_step_device_dispatch(
        device, _vae_train_step!, loader, opt_state)

    if print_log; println("Training VAE... (β = $β)") end
    for epoch in 1:epochs
        epoch_loss = T(0.0)

        if epoch % 100 == 0 && shuffle
            loader = load_data(T.(protocol.training_data), batchsize, shuffle) |> device
        end

        for x_batch in loader
            elbo, opt_state, rng = _train_step!(x_batch, opt_state, rng)
            epoch_loss += elbo
        end

        mean_elbo = -epoch_loss / length(loader)
        protocol._log["Mean ELBO"][epoch] = mean_elbo

        if print_log && (epoch % 5 == 0 || epoch == 1)
            @printf("Epoch %8d | Avg. ELBO: %16.8e \n", epoch, mean_elbo)
        end
    end
    if print_log; println("Training complete!") end

    model._ps[] = opt_state.parameters |> cpu_device()
    model._st[] = opt_state.states |> cpu_device()

    return protocol._log
end

function _vae_encode(encoders, ps, st, x, rng = Random.default_rng())
    num_latent_layers = length(encoders)
    latent_size = round(Int, _output_size(encoders[end]) / 2)
    T = eltype(x)

    enc_out, st_new = encoders[1](x, ps.encoders[1], st.encoders[1])
    st_encoders_list = Any[st.encoders...]
    st_encoders_list[1] = st_new
    
    mu = enc_out[1:latent_size, :]
    logvar = enc_out[(latent_size + 1):end, :]
    z, rng = sample_latent(mu, exp.(logvar .* T(0.5)), rng)
    
    for i in 2:num_latent_layers
        enc_out, st_new = encoders[i](z, ps.encoders[i], st.encoders[i])
        st_encoders_list[i] = st_new
        
        mu = enc_out[1:latent_size, :]
        logvar = enc_out[(latent_size + 1):end, :]
        z, rng = sample_latent(mu, exp.(logvar .* T(0.5)), rng)
    end

    return z, rng
end

function _encode(model::VariationalAutoencoder, x::Matrix)
    T = eltype(model._ps[].encoders[1].layer_1.weight)
    return first(_vae_encode(model.encoders, model._ps[], model._st[], T.(x)))
end

function _encode(model::VariationalAutoencoder, x::Vector)
    return _encode(model, reshape(x, :, 1))[:]
end

function _vae_decode(decoders, ps, st, z, rng = Random.default_rng())
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
        z, rng = sample_latent(μ, exp.(log_σ² .* T(0.5)), rng)
    end

    out, st_new = decoders[1](z, ps.decoders[1], st.decoders[1])
    st_decoders_list[1] = st_new

    return out, rng
end

function _decode(model::VariationalAutoencoder, z::Matrix)
    T = eltype(model._ps[].decoders[1].layer_1.weight)
    return first(_vae_decode(model.decoders, model._ps[], model._st[], T.(z)))
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

