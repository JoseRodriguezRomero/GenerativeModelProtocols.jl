macro _gan_default_activation_function()
    return elu
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

$TYPEDFIELDS
"""
@kwdef struct GenerativeAdversarialNetwork <: AbstractCategoricalGenerativeModel
    """Neural network discriminating between real and synthetic data."""
    discriminator::Chain
    """Variational Autoencoder for encoding and decoding."""
    vae_model::VariationalAutoencoder
    """Trained parameters of the discriminator model. Users should not use directly use this."""
    _ps_discriminator::Union{Ref{<:NamedTuple}, Nothing} = nothing
    """Trained state of the discriminator model. Users should not use directly use this."""
    _st_discriminator::Union{Ref{<:NamedTuple}, Nothing} = nothing

    function GenerativeAdversarialNetwork(
    discriminator::Chain, 
    vae_model::VariationalAutoencoder,
    _ps_discriminator::Union{Ref{<:NamedTuple}, Nothing},
    _st_discriminator::Union{Ref{<:NamedTuple}, Nothing}
    )

        if !compatible_gan_model(discriminator, vae_model)
            @error "Incompatible GenerativeAdversarialNetwork architecture!"
            throw(MethodError(GenerativeAdversarialNetwork, (discriminator, vae_model)))
        end

        if isnothing(_ps_discriminator) && isnothing(_st_discriminator)
            _ps_discriminator_val, _st_discriminator_val = Lux.setup(Random.default_rng(), (discriminator = discriminator,))
            _ps_discriminator = Ref{NamedTuple}(_ps_discriminator_val)
            _st_discriminator = Ref{NamedTuple}(_st_discriminator_val)
        end

        return new(discriminator, vae_model, _ps_discriminator, _st_discriminator)
    end
end

"""
    GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size::Int, latent_size::Int)

Convenience constructor that creates a 
`GenerativeModelProtocols.GenerativeAdversarialNetwork` using default generator
and discriminator networks.
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
    return first(discriminator(x, ps.discriminator, st.discriminator))
end

function _gan_grad_penalty(discriminator, real_data, fake_data, a::AbstractFloat, ps, st)
    T = eltype(real_data)
    ϵ = rand_like(real_data, size(real_data))
    interpolates = ϵ .* real_data .+ (T(1.0) .- ϵ) .* fake_data

    stateful_discriminator = Lux.StatefulLuxLayer(discriminator, ps.discriminator, st.discriminator)

    grads, = Enzyme.gradient(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        (x, sd) -> sum(sd(x)),
        interpolates,
        Enzyme.Const(stateful_discriminator)
    )
    
    grad_norms = sqrt.(sum(abs2, grads, dims=1) .+ T(1.0E-8))
    return mean(abs2.(grad_norms .- a))
end

function _gan_make_fake_recon_data(encoders, decoders, latent_size, ps_vae, st_vae, real_data)
    ẑ = randn_like(real_data, (latent_size, size(real_data, 2)))
    z = _vae_encode(encoders, ps_vae, st_vae, real_data)
    
    fake_data = _vae_decode(decoders, ps_vae, st_vae, ẑ)
    recon_data = _vae_decode(decoders, ps_vae, st_vae, z)

    return fake_data, recon_data
end

function _gan_disc_loss(
    discriminator, ps_disc, st_disc, 
    real_data, fake_data, recon_data,
    grad_penalty::Bool, λ::AbstractFloat, a::AbstractFloat)
    
    T = eltype(real_data)

    disc_real_data = _gan_discriminate(discriminator, real_data, ps_disc, st_disc)
    disc_fake_data = _gan_discriminate(discriminator, fake_data, ps_disc, st_disc)
    disc_recon_data = _gan_discriminate(discriminator, recon_data, ps_disc, st_disc)
    
    w_loss = T(0.5) .* (mean(disc_fake_data) + mean(disc_recon_data)) - mean(disc_real_data)

    if grad_penalty
        fake_data_grad_penalty = _gan_grad_penalty(discriminator, real_data, fake_data, a, ps_disc, st_disc)
        recon_data_grad_penalty = _gan_grad_penalty(discriminator, real_data, recon_data, a, ps_disc, st_disc)

        return w_loss + T(0.5) * λ * (fake_data_grad_penalty + recon_data_grad_penalty)
    end

    return w_loss
end

function _gan_vae_loss(
    encoders, decoders, discriminator, 
    ps_disc, st_disc, ps_vae, st_vae, 
    latent_size, num_latent_layers, batch_size, 
    real_data, fake_data, recon_data,
    β::AbstractFloat, γ_vae::AbstractFloat, γ_wgan::AbstractFloat)

    T = eltype(real_data)

    disc_fake_data = _gan_discriminate(discriminator, fake_data, ps_disc, st_disc)
    disc_recon_data = _gan_discriminate(discriminator, recon_data, ps_disc, st_disc)

    vae_elbo = _vae_elbo(encoders, decoders, β, latent_size, num_latent_layers, batch_size, real_data, ps_vae, st_vae)
    wgan_loss = T(-0.5) .* (mean(disc_fake_data) + mean(disc_recon_data))

    return γ_vae * vae_elbo + γ_wgan * wgan_loss
end

function _train!(protocol::GenerativeModelProtocol, model::GenerativeAdversarialNetwork, 
    β::AbstractFloat, γ_vae::AbstractFloat, γ_wgan::AbstractFloat; 
    print_log::Bool, n_critic::Int, grad_penalty::Bool, λ::AbstractFloat, a::AbstractFloat,
    weight_clipping::Bool, clip_value::AbstractFloat)

    if !grad_penalty && !weight_clipping
        @warn "Training a WGAN with neither gradient penalty nor weight clipping active is not recommended."
    elseif grad_penalty && weight_clipping
        @warn "Training a WGAN with gradient penalty and weight clipping simultaneously active is not recommended."
    end

    ps_vae = protocol.precision(model.vae_model._ps[]) |> protocol.device
    st_vae = protocol.precision(model.vae_model._st[]) |> protocol.device

    ps_disc = protocol.precision(model._ps_discriminator[]) |> protocol.device
    st_disc = protocol.precision(model._st_discriminator[]) |> protocol.device

    training_data_device = protocol.training_data |> protocol.device
    shuffle_device = protocol.shuffle

    T = eltype(protocol.training_data)
    β_device = T(β) |> protocol.device
    λ_device = T(λ) |> protocol.device
    a_device = T(a) |> protocol.device
    clip_value_device = T(clip_value) |> protocol.device
    γ_vae_device = T(γ_vae) |> protocol.device
    γ_wgan_device = T(γ_wgan) |> protocol.device
    grad_penalty_device = grad_penalty |> protocol.device
    weight_clipping_device = weight_clipping |> protocol.device
    n_critic_device = n_critic |> protocol.device
    latent_size_device = _latent_size(model) |> protocol.device
    num_latent_layers_device = length(model.vae_model.encoders) |> protocol.device

    encoders = model.vae_model.encoders
    decoders = model.vae_model.decoders
    discriminator = model.discriminator

    opt_disc = nothing
    opt_vae_model = nothing

    if isa(protocol.optimiser, Tuple)
        opt_disc = deepcopy(protocol.optimiser[1])
        opt_vae_model = deepcopy(protocol.optimiser[2])
    else
        opt_disc = deepcopy(protocol.optimiser)
        opt_vae_model = deepcopy(protocol.optimiser)
    end

    loader = load_data(training_data_device, protocol.batchsize, shuffle_device)

    function _disc_train_step!(real_data, p_current, s_current, o_current)
        local loss_val, p_updated, o_updated = T(0.0), p_current, o_current
        fake_data, recon_data = _gan_make_fake_recon_data(encoders, decoders, latent_size_device, ps_vae, st_vae, real_data)

        for i in 1:n_critic_device
            _objective = (p) -> begin
                return _gan_disc_loss(
                    discriminator, p, s_current, 
                    real_data, fake_data, recon_data, 
                    (i == n_critic_device) ? grad_penalty_device : false, λ_device, a_device
                )
            end

            loss_val = _objective(p_updated)
            loss_grads = Enzyme.make_zero(p_updated)

            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const(_objective),
                Enzyme.Active,
                Enzyme.Duplicated(p_updated, loss_grads)
            )

            o_updated, p_updated = Optimisers.update(o_updated, p_updated, loss_grads)

            if weight_clipping_device
                c_val = T(clip_value_device)
                p_inner = p_updated.discriminator
                p_clamped_inner = NamedTuple{keys(p_inner)}(
                    ntuple(i -> (
                        weight = clamp.(p_inner[i].weight, -c_val, c_val),
                        bias = clamp.(p_inner[i].bias, -c_val, c_val)
                    ), Val(length(p_inner)))
                )
                p_updated = (discriminator = p_clamped_inner,)
            end
        end

        return loss_val, p_updated, o_updated
    end

    function _vae_train_step!(real_data, p_current, s_current, o_current)
        fake_data, recon_data = _gan_make_fake_recon_data(encoders, decoders, latent_size_device, ps_vae, st_vae, real_data)

        _objective = (p) -> _gan_vae_loss(
            encoders, decoders, discriminator, 
            ps_disc, st_disc, p, s_current, 
            latent_size_device, num_latent_layers_device, size(real_data,2), 
            real_data, fake_data, recon_data, 
            β_device, γ_vae_device, γ_wgan_device
        )
        
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

    opt_state_disc = _initial_step(model.discriminator, ps_disc, st_disc, opt_disc)
    opt_state_vae = _initial_step(model.vae_model, ps_vae, st_vae, opt_vae_model)

    _train_step_disc!, opt_state_disc = _train_step_device_dispatch(protocol.device, _disc_train_step!, loader, opt_state_disc)
    _train_step_vae!, opt_state_vae = _train_step_device_dispatch(protocol.device, _vae_train_step!, loader, opt_state_vae)

    for epoch in 1:protocol.epochs
        if epoch % 100 == 0 && protocol.shuffle
            loader = load_data(training_data_device, protocol.batchsize, shuffle_device)
        end

        running_loss_critic = T(0.0)
        running_loss_vae = T(0.0)

        for real_data in loader
            loss_critic, opt_state_disc = _train_step_disc!(real_data, opt_state_disc)
            loss_vae, opt_state_vae = _train_step_vae!(real_data, opt_state_vae)
            
            running_loss_critic += loss_critic
            running_loss_vae += loss_vae
        end

        protocol._log.loss[epoch] = running_loss_critic / length(loader)
        protocol._log.loss_grad_norm[epoch] = running_loss_vae / length(loader)
        
        if print_log && (epoch % 5 == 0 || epoch == 1)
            avg_loss_critic = running_loss_critic / length(loader)
            avg_loss_vae = running_loss_vae / length(loader)
        
            @printf("Epoch %5d", epoch)
            @printf(" | Avg. Critic Loss: %15.6E", avg_loss_critic)
            @printf(" | Avg. VAE Loss: %15.6E \n", avg_loss_vae)
        end
    end

    model.vae_model._ps[] = opt_state_vae.parameters
    model.vae_model._st[] = opt_state_vae.states

    model._ps_discriminator[] = opt_state_disc.parameters
    model._st_discriminator[] = opt_state_disc.states

    return protocol._log
end

function _train!(protocol::GenerativeModelProtocol, model::GenerativeAdversarialNetwork; 
    print_log::Bool = true, n_critic::Int = 5, 
    β::Union{AbstractFloat, Vector{<:AbstractFloat}} = 1.0,
    γ_vae::AbstractFloat = 1.0, γ_wgan::AbstractFloat = 1.0,
    grad_penalty::Bool = false, λ::AbstractFloat = 10.0, a::AbstractFloat = 1.0,
    weight_clipping::Bool = false, clip_value::AbstractFloat = 1.0)

    for i in eachindex(β)
        _train!(protocol, model, β[i], γ_vae, γ_wgan; 
            print_log = print_log, n_critic = n_critic,
            grad_penalty = grad_penalty, λ = λ, a = a, 
            weight_clipping = weight_clipping, clip_value = clip_value
        )
    end

    return protocol._log
end

function _eval(model::GenerativeAdversarialNetwork, n_samples::Int)
    return _eval(model.vae_model, n_samples)
end

function _eval(model::GenerativeAdversarialNetwork)
    return _eval(model, 1)[:]
end

function _categorize(model::GenerativeAdversarialNetwork, x::AbstractMatrix)::AbstractMatrix
    return model.discriminator_network(x)
end

function _categorize(model::GenerativeAdversarialNetwork, x::AbstractVector)::AbstractVector
    return _categorize(model, reshape(x, :, 1))[:]
end

function _encode(model::GenerativeAdversarialNetwork, x::AbstractMatrix)::AbstractMatrix
    return _encode(model.vae_model, x)
end

function _encode(model::GenerativeAdversarialNetwork, x::AbstractVector)::AbstractVector
    return _encode(model, reshape(x, :, 1))[:]
end

function _decode(model::GenerativeAdversarialNetwork, z::AbstractMatrix)::AbstractMatrix
    return _decode(model.vae_model, z)
end

function _decode(model::GenerativeAdversarialNetwork, z::AbstractVector)::AbstractVector
    return _decode(model, reshape(z, :, 1))[:]
end

