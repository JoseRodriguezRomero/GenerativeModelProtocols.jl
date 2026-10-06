macro _gan_default_activation_function()
    return swish
end

function _gan_default_discriminator_network(input_size::Int; hidden_layer_size::Int = 32, activation_function::Function = @_gan_default_activation_function)
    return Chain(
        Dense(input_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 1)
    )
end

function _gan_default_vae_model(input_size::Int, latent_dim::Int, latent_layers::Int; hidden_layer_size::Int = 32, activation_function::Function = @_gan_default_activation_function)
    default_encoders = _vae_default_encoder_network(input_size, latent_dim, latent_layers, hidden_layer_size, activation_function)
    default_decoders = _vae_default_decoder_network(input_size, latent_dim, latent_layers, hidden_layer_size, activation_function)
    
    return VariationalAutoencoder(default_encoders, default_decoders)
end

function compatible_gan_model(discriminator::C, vae_model::VariationalAutoencoder)::Bool where {C<:Chain}
    if _input_size(vae_model) != _input_size(discriminator)
        return false
    end

    if _output_size(discriminator) != 1
        return false
    end

    return true
end

@compat public GenerativeAdversarialNetwork

"""
$TYPEDEF

A structure containing the general parameters needed to evaluate and train a 
Generative Adversarial Network (GAN). Once trained, it can be used a generative 
model.

# Structure Fields
$TYPEDFIELDS
"""
@kwdef struct GenerativeAdversarialNetwork <: AbstractCategoricalGenerativeModel
    """Neural network discriminating between real and synthetic data."""
    discriminator::Chain
    """Variational Autoencoder for encoding and decoding."""
    vae_model::VariationalAutoencoder
    """Trained parameters of the discriminator model. Users should not use directly use this."""
    _ps_discriminator::Ref{<:NamedTuple} = Ref{NamedTuple}(NamedTuple())
    """Trained state of the discriminator model. Users should not use directly use this."""
    _st_discriminator::Ref{<:NamedTuple} = Ref{NamedTuple}(NamedTuple())

    function GenerativeAdversarialNetwork(
    discriminator::Chain, 
    vae_model::VariationalAutoencoder,
    _ps_discriminator::Ref{<:NamedTuple},
    _st_discriminator::Ref{<:NamedTuple}
    )

        if !compatible_gan_model(discriminator, vae_model)
            @error "Incompatible GenerativeAdversarialNetwork architecture!"
            throw(MethodError(GenerativeAdversarialNetwork, (discriminator, vae_model)))
        end

        if isempty(_ps_discriminator[]) || isempty(_st_discriminator[])
            _ps_discriminator_val, _st_discriminator_val = Lux.setup(Random.default_rng(), (discriminator = discriminator,))
            _ps_discriminator[] = _ps_discriminator_val
            _st_discriminator[] = _st_discriminator_val
        end

        return new(discriminator, vae_model, _ps_discriminator, _st_discriminator)
    end
end

"""
    GenerativeModelProtocols.GenerativeAdversarialNetwork(
        input_size::Int, latent_size::Int = 1, latent_layers::Int = 1)

Convenience constructor that creates a 
`GenerativeModelProtocols.GenerativeAdversarialNetwork` using default generator
and discriminator networks.

# Arguments
 - `input_size::Int`: The dimensionality of the input data.
 - `latent_size::Int`: The size of the latent space (default is 1).
 - `latent_layers::Int`: The number of layers in the latent space (default is 1).
"""
function GenerativeAdversarialNetwork(input_size::Int, latent_size::Int = 1, latent_layers::Int = 1)
    return GenerativeAdversarialNetwork(;
        discriminator = _gan_default_discriminator_network(input_size),
        vae_model     = _gan_default_vae_model(input_size, latent_size, latent_layers)
    )
end

function load_generative_adversarial_network_parameters end

macro load_generative_adversarial_network_parameters(saved_model, main_group_name, generative_model_group_name)
    return :(load_generative_adversarial_network_parameters($(esc(saved_model));
        $(main_group_name = esc(main_group_name)),
        $(generative_model_group_name = esc(generative_model_group_name))
    ))
end

function GenerativeAdversarialNetwork(saved_model::String; 
    main_group_name = @default_main_group_name,
    generative_model_group_name = @default_generative_model_group_name)

    return @load_generative_adversarial_network_parameters(saved_model, main_group_name, generative_model_group_name)
end

function _default_optimiser(::GenerativeAdversarialNetwork)
    critic_optimiser = Adam(; eta = 1.0E-4, beta = (0.0,0.9))
    vae_optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999))
    return (critic_optimiser, vae_optimiser)
end

function Base.display(model::GenerativeAdversarialNetwork)
    print_padding = @_default_print_padding
    println("$(summary(model)):")

    println("discriminator: ")
    _print_chains((discriminator = model.discriminator,), print_padding)

    println("")

    println("encoders: ")
    _print_chains(model.vae_model.encoders, print_padding)

    println("")

    println("decoders: ")
    _print_chains(model.vae_model.decoders, print_padding)
end

function _input_size(model::GenerativeAdversarialNetwork)::Int
    return _input_size(model.vae_model)
end

function _latent_size(model::GenerativeAdversarialNetwork)::Int
    return _latent_size(model.vae_model)
end 

function _generative_model(::GenerativeAdversarialNetwork)::GenerativeModel
    return generative_adversarial_network
end

function _gan_discriminate(discriminator, x, ps, st)
    disc_val, st_val = discriminator(x, ps.discriminator, st.discriminator)
    return disc_val, (discriminator = st_val,)
end

function _gan_penalty_model(discriminator::Chain)
    layers = ()

    for layer in values(discriminator.layers)

        if layer isa Dense && layer.activation !== identity
            linear_layer = Dense(
                layer.in_dims => layer.out_dims,
                identity;
                use_bias=layer.use_bias
            )
            activation_layer = Lux.WrappedFunction(
                Base.Fix1(broadcast, layer.activation)
            )

            layers = (layers..., linear_layer, activation_layer)
        else
            layers = (layers..., layer)
        end
    end

    return Chain(layers...)
end

function _gan_penalty_parameters(discriminator::Chain, penalty_model::Chain, ps, st)
    layer_ps = ()
    layer_st = ()

    for (layer_name, layer) in zip(keys(discriminator.layers), values(discriminator.layers))
        ps_layer = getproperty(ps.discriminator, layer_name)
        st_layer = getproperty(st.discriminator, layer_name)

        if layer isa Dense && layer.activation !== identity
            layer_ps = (layer_ps..., ps_layer, NamedTuple())
            layer_st = (layer_st..., st_layer, NamedTuple())
        else
            layer_ps = (layer_ps..., ps_layer)
            layer_st = (layer_st..., st_layer)
        end
    end

    return (
        NamedTuple{keys(penalty_model.layers)}(layer_ps),
        NamedTuple{keys(penalty_model.layers)}(layer_st)
    )
end

function _gan_grad_penalty(
    discriminator, penalty_model, real_data, fake_data, a::AbstractFloat, ps, st, ϵ
)
    T = eltype(real_data)
    interpolates = ϵ .* real_data .+ (T(1.0) .- ϵ) .* fake_data

    penalty_ps, penalty_st = _gan_penalty_parameters(discriminator, penalty_model, ps, st)
    stateful_discriminator = Lux.StatefulLuxLayer(
        penalty_model,
        penalty_ps,
        penalty_st
    )
    objective = x -> sum(stateful_discriminator(x))

    grads, = Enzyme.gradient(
        Enzyme.Reverse,
        Enzyme.Const(objective),
        interpolates
    )

    grad_norms = sqrt.(sum(abs2, grads, dims=1) .+ T(1.0E-8))
    _, st = _gan_discriminate(discriminator, interpolates, ps, st)

    penalty_elements = abs2.(grad_norms .- a)
    finite_mask = isfinite.(penalty_elements)
    finite_penalty_sum = sum(ifelse.(finite_mask, penalty_elements, zero(T)))
    finite_penalty_count = sum(finite_mask)
    penalty = finite_penalty_sum / finite_penalty_count

    return penalty, st
end

function _gan_make_fake_recon_data(encoders, decoders, ps_vae, st_vae, real_data, rng)
    z, rng = _vae_encode(encoders, ps_vae, st_vae, real_data, rng)
    recon_data, rng = _vae_decode(decoders, ps_vae, st_vae, z, rng)

    ẑ = similar(z)
    randn!(rng, ẑ)
    fake_data, rng = _vae_decode(decoders, ps_vae, st_vae, ẑ, rng)
    
    return fake_data, recon_data, rng
end

function _gan_disc_loss(
    discriminator, penalty_model, ps_disc, st_disc,
    real_data, fake_data, recon_data,
    grad_penalty::Bool, λ::AbstractFloat, a::AbstractFloat, ϵ)
    
    T = eltype(real_data)

    disc_real_data, st_disc = _gan_discriminate(discriminator, real_data, ps_disc, st_disc)
    disc_fake_data, st_disc = _gan_discriminate(discriminator, fake_data, ps_disc, st_disc)
    disc_recon_data, st_disc = _gan_discriminate(discriminator, recon_data, ps_disc, st_disc)
    
    w_loss = T(0.5) .* (mean(disc_fake_data) + mean(disc_recon_data)) - mean(disc_real_data)

    if grad_penalty
        fake_data_grad_penalty, st_disc = _gan_grad_penalty(
            discriminator, penalty_model, real_data, fake_data, a, ps_disc, st_disc, ϵ
        )
        recon_data_grad_penalty, st_disc = _gan_grad_penalty(
            discriminator, penalty_model, real_data, recon_data, a, ps_disc, st_disc, ϵ
        )

        w_loss += T(0.5) * λ * (fake_data_grad_penalty + recon_data_grad_penalty)
    end

    return w_loss, st_disc
end

function _gan_vae_loss(
    encode_vae, decode_vae, ps_vae, st_vae,
    real_data, latent_data, wgan_loss,
    β::AbstractFloat, γ_vae::AbstractFloat, γ_wgan::AbstractFloat)

    vae_elbo, st_vae = _vae_elbo(
        encode_vae, decode_vae, β, real_data, latent_data, ps_vae, st_vae
    )
    loss = γ_vae * vae_elbo + γ_wgan * wgan_loss

    return loss, st_vae
end

function _train!(protocol::GenerativeModelProtocol, model::GenerativeAdversarialNetwork;
    n_critic::Int = 5, β::AbstractFloat = 1.0, γ_vae::AbstractFloat = 1.0, 
    γ_wgan::AbstractFloat = 1.0, grad_penalty::Bool = false, λ::AbstractFloat = 10.0, 
    a::AbstractFloat = 1.0, weight_clipping::Bool = false, clip_value::AbstractFloat = 1.0,
    print_log::Bool, epochs::Int, batchsize::Int, shuffle::Bool, 
    optimiser::Tuple{<:AbstractRule, <:AbstractRule}, 
    device::MLDataDevices.AbstractDevice, precision::Function)

    if !grad_penalty && !weight_clipping
        @warn "Training a WGAN with neither gradient penalty nor weight clipping active is not recommended."
    elseif grad_penalty && weight_clipping
        @warn "Training a WGAN with gradient penalty and weight clipping simultaneously active is not recommended."
    end

    ps_vae = precision(model.vae_model._ps[]) |> device
    st_vae = precision(model.vae_model._st[]) |> device

    ps_disc = precision(model._ps_discriminator[]) |> device
    st_disc = precision(model._st_discriminator[]) |> device

    T = eltype(precision([1.0]))
    β_device = T(β) |> device
    λ_device = T(λ) |> device
    a_device = T(a) |> device
    clip_value_device = T(clip_value) |> device
    γ_vae_device = T(γ_vae) |> device
    γ_wgan_device = T(γ_wgan) |> device
    grad_penalty_device = grad_penalty |> device
    weight_clipping_device = weight_clipping |> device
    n_critic_device = n_critic |> device
    latent_dim = _latent_size(model)
    num_latent_layers = length(model.vae_model.encoders)

    encoders = model.vae_model.encoders
    decoders = model.vae_model.decoders
    discriminator = model.discriminator

    protocol._log["Mean Critic Loss"] = zeros(T, epochs)
    protocol._log["Mean VAE Loss"] = zeros(T, epochs)

    opt_disc = optimiser[1]
    opt_vae_model = optimiser[2]

    loader = load_data(T.(protocol.training_data), batchsize, shuffle) |> device
    real_data_init = first(loader)
    latent_data_init = ntuple(
        _ -> fill!(
            similar(real_data_init, T, latent_dim, size(real_data_init, 2)), zero(T)
        ),
        num_latent_layers
    )
    
    ϵ_init = ntuple(
        _ -> fill!(
            similar(real_data_init, T, latent_dim, size(real_data_init, 2)), zero(T)
        ),
        num_latent_layers
    )

    make_latent_variables = _function_device_dispatch(
        device, _make_vae_latent_variables(encoders, latent_dim, num_latent_layers),
        real_data_init, ϵ_init, ps_vae, st_vae
    )

    encode_vae = _function_device_dispatch(
        device, _encode_vae(encoders, latent_dim),
        real_data_init, latent_data_init, ps_vae.encoders, st_vae.encoders
    )

    penalty_discriminator = _gan_penalty_model(discriminator)

    decode_vae = _function_device_dispatch(
        device, _decode_vae(decoders, latent_dim),
        latent_data_init, ps_vae.decoders, st_vae.decoders
    )

    function _disc_train_step!(real_data, p_current, s_current, o_current, rng)
        local loss_val, p_updated, s_updated, o_updated = T(0.0), p_current, s_current, o_current

        rng_trace = Lux.replicate(rng)
        ϵ = similar(real_data)
        rand!(rng_trace, ϵ)

        for i in 1:n_critic_device
            do_grad_penalty = (i == n_critic_device) ? grad_penalty_device : false

            fake_data, recon_data, rng_trace = _gan_make_fake_recon_data(
                encoders, decoders, ps_vae, st_vae, real_data, rng_trace
            )

            _objective = (ps_disc, st_disc) -> _gan_disc_loss(
                discriminator, penalty_discriminator, ps_disc, st_disc,
                real_data, fake_data, recon_data,
                do_grad_penalty, λ_device, a_device, ϵ
            )

            loss_grads = Enzyme.make_zero(p_updated)

            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const((ps_disc, st_disc) -> _objective(ps_disc, st_disc)[1]),
                Enzyme.Active,
                Enzyme.Duplicated(p_updated, loss_grads),
                Enzyme.Const(s_updated)
            ) # crash happens here

            loss_val, s_updated = _objective(p_updated, s_current)
            o_updated, p_updated = Optimisers.update(o_updated, p_updated, loss_grads)

            if weight_clipping_device
                c_val = T(clip_value_device)
                
                p_updated_disc = map(p_updated.discriminator) do layer
                    (
                        weight = clamp.(layer.weight, -c_val, c_val),
                        bias = clamp.(layer.bias, -c_val, c_val)
                    )
                end

                p_updated = (discriminator = p_updated_disc,)
            end
        end

        return loss_val, p_updated, s_updated, o_updated, rng_trace
    end

    function _vae_train_step!(real_data, p_current, s_current, o_current, rng)
        rng_trace = Lux.replicate(rng)
        fake_data, recon_data, rng_trace = _gan_make_fake_recon_data(
            encoders, decoders, ps_vae, st_vae, real_data, rng_trace
        )

        batch_size = size(real_data, 2)
        ϵ, rng_trace = _sample_vae_latent_variables(
            real_data, latent_dim, num_latent_layers, batch_size, rng_trace
        )

        disc_fake_data, _ = _gan_discriminate(
            discriminator, fake_data, ps_disc, st_disc
        )

        disc_recon_data, _ = _gan_discriminate(
            discriminator, recon_data, ps_disc, st_disc
        )
        
        wgan_loss = T(-0.5) .* (mean(disc_fake_data) + mean(disc_recon_data))

        _objective = (ps_vae, st_vae) -> begin
            latent_data = make_latent_variables(real_data, ϵ, ps_vae, st_vae)

            return _gan_vae_loss(
                encode_vae, decode_vae, ps_vae, st_vae, real_data, latent_data,
                wgan_loss, β_device, γ_vae_device, γ_wgan_device
            )
        end
        
        loss_grads = Enzyme.make_zero(p_current)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const((p, s) -> _objective(p, s)[1]),
            Enzyme.Active,
            Enzyme.Duplicated(p_current, loss_grads),
            Enzyme.Const(s_current)
        )

        loss_val, s_updated = _objective(p_current, s_current)
        o_updated, p_updated = Optimisers.update(o_current, p_current, loss_grads)

        return loss_val, p_updated, s_updated, o_updated, rng_trace
    end

    opt_state_disc = _initial_step(model.discriminator, ps_disc, st_disc, opt_disc)
    opt_state_vae = _initial_step(model.vae_model, ps_vae, st_vae, opt_vae_model)

    _train_step_disc!, opt_state_disc, _ = _train_step_device_dispatch(device, _disc_train_step!, loader, opt_state_disc)
    _train_step_vae!, opt_state_vae, rng = _train_step_device_dispatch(device, _vae_train_step!, loader, opt_state_vae)

    for epoch in 1:epochs
        if epoch % 100 == 0 && shuffle
            loader = load_data(T.(protocol.training_data), batchsize, shuffle) |> device
        end

        running_loss_critic = T(0.0)
        running_loss_vae = T(0.0)

        for real_data in loader
            loss_critic, opt_state_disc, rng = _train_step_disc!(real_data, opt_state_disc, rng)
            loss_vae, opt_state_vae, rng = _train_step_vae!(real_data, opt_state_vae, rng)
            
            running_loss_critic += loss_critic
            running_loss_vae += loss_vae
        end

        avg_loss_critic = running_loss_critic / length(loader)
        avg_loss_vae = running_loss_vae / length(loader)

        protocol._log["Mean Critic Loss"][epoch] = avg_loss_critic
        protocol._log["Mean VAE Loss"][epoch] = avg_loss_vae
        
        if print_log && (epoch % 5 == 0 || epoch == 1)    
            @printf("Epoch %5d", epoch)
            @printf(" | Avg. Critic Loss: %15.6E", avg_loss_critic)
            @printf(" | Avg. VAE Loss: %15.6E \n", avg_loss_vae)
        end
    end

    model.vae_model._ps[] = opt_state_vae.parameters |> cpu_device()
    model.vae_model._st[] = opt_state_vae.states |> cpu_device()

    model._ps_discriminator[] = opt_state_disc.parameters |> cpu_device()
    model._st_discriminator[] = opt_state_disc.states |> cpu_device()

    return protocol._log
end

function _eval(model::GenerativeAdversarialNetwork, n_samples::Int)
    return _eval(model.vae_model, n_samples)
end

function _eval(model::GenerativeAdversarialNetwork)
    return _eval(model, 1)[:]
end

function _categorize(model::GenerativeAdversarialNetwork, x::Matrix)::Matrix
    T = eltype(model._ps_discriminator[].discriminator.weight)
    return model.discriminator_network(T.(x))
end

function _categorize(model::GenerativeAdversarialNetwork, x::Vector)::Vector
    T = eltype(model._ps_discriminator[].discriminator.weight)
    return _categorize(model, reshape(T.(x), :, 1))[:]
end

function _encode(model::GenerativeAdversarialNetwork, x::Matrix)::Matrix
    T = eltype(model.vae_model._ps[].encoders[1].layer_1.weight)
    return _encode(model.vae_model, T.(x))
end

function _encode(model::GenerativeAdversarialNetwork, x::Vector)::Vector
    T = eltype(model.vae_model._ps[].encoders[1].layer_1.weight)
    return _encode(model, reshape(T.(x), :, 1))[:]
end

function _decode(model::GenerativeAdversarialNetwork, z::Matrix)::Matrix
    T = eltype(model.vae_model._ps[].decoders[1].layer_1.weight)
    return _decode(model.vae_model, T.(z))
end

function _decode(model::GenerativeAdversarialNetwork, z::Vector)::Vector
    T = eltype(model.vae_model._ps[].decoders[1].layer_1.weight)
    return _decode(model, reshape(T.(z), :, 1))[:]
end
