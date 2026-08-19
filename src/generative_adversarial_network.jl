macro _gan_default_activation_function()
    return leakyrelu
end

function _gan_default_discriminator_network(input_size::Int; hidden_layer_size::Int = 32, activation_function::Function = @_gan_default_activation_function)
    return Chain(
        Dense(input_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 1)
    ) |> f64
end

function _gan_default_vae_model(input_size::Int, latent_dim::Int, latent_layers::Int; hidden_layer_size::Int = 32, activation_function::Function = @_gan_default_activation_function)
    default_encoders = _vae_default_encoder_network(input_size, latent_dim, latent_layers, hidden_layer_size, activation_function)
    default_decoders = _vae_default_decoder_network(input_size, latent_dim, latent_layers, hidden_layer_size, activation_function)
    
    return VariationalAutoencoder(default_encoders, default_decoders)
end

function compatible_gan_model(discriminator::Chain, vae_model::VariationalAutoencoder)::Bool
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
    
    function GenerativeAdversarialNetwork(discriminator::Chain, vae_model::VariationalAutoencoder)
        if !compatible_gan_model(discriminator, vae_model)
            @error "Incompatible GenerativeAdversarialNetwork architecture!"
            throw(MethodError(GenerativeAdversarialNetwork, (discriminator, vae_model)))
        end

        return new(discriminator, vae_model)
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
    _print_chains(model.discriminator, print_padding)

    println("")

    println("encoder: ")
    _print_chains(model.vae_model.encoders, print_padding)

    println("")

    println("decoder: ")
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

function _gan_make_latent_variables(model::GenerativeAdversarialNetwork, real_data)
    return Flux.randn_like(real_data, (_latent_size(model.vae_model), size(real_data, 2)))
end

function _gan_grad_penalty(discriminator_net, real_data, fake_data, λ::Float64, a::Float64)
    ϵ = Flux.rand_like(real_data, size(real_data))
    interpolates = ϵ .* real_data .+ (1.0 .- ϵ) .* fake_data

    features, _ = size(interpolates)
    net_tangents = ntuple(_ -> Enzyme.make_zero(discriminator_net), Val(features))

    seed_tangents = ntuple(features) do i
        t = zero(interpolates)
        t[i, :] .= 1.0
        return t
    end

    d_out = Enzyme.autodiff(
        Enzyme.Forward,
        Enzyme.BatchDuplicated(discriminator_net, net_tangents),
        Enzyme.BatchDuplicated(interpolates, seed_tangents)
    )

    grads_interp = similar(interpolates)
    for i in 1:features
        grads_interp[i, :] .= vec(d_out[1][i])
    end

    grad_norms = sqrt.(sum(abs2, grads_interp, dims=1) .+ 1.0E-8)
    gp = mean(abs2.(grad_norms .- a))

    return λ * gp
end

function _train_discriminator!(model::GenerativeAdversarialNetwork, real_data, opt_state_discriminator; 
    grad_penalty::Bool, λ::Float64, a::Float64, weight_clipping::Bool, clip_value::Float64)
    
    fake_data = _decode(model.vae_model, _gan_make_latent_variables(model, real_data))
    recon_data = _decode(model.vae_model, _encode(model.vae_model, real_data))

    loss_c, grads_crit = Flux.withgradient(AutoEnzyme(), model.discriminator) do discriminator_net
        disc_real = discriminator_net(real_data)
        disc_fake = discriminator_net(fake_data)
        disc_recon = discriminator_net(recon_data)
        w_loss =  0.5 .* (mean(disc_fake) + mean(disc_recon)) - mean(disc_real)

        if grad_penalty
            fake_data_grad_penalty = _gan_grad_penalty(discriminator_net, real_data, fake_data, λ, a)
            recon_data_grad_penalty = _gan_grad_penalty(discriminator_net, real_data, recon_data, λ, a)
            return w_loss + 0.5 * (fake_data_grad_penalty + recon_data_grad_penalty)
        else
            return w_loss
        end
    end

    if weight_clipping
        foreach(Flux.trainable(model.discriminator)) do layer_params
            foreach(layer_params) do p
                if p isa AbstractArray
                    p .= clamp.(p, -clip_value, clip_value)
                end
            end
        end
    end

    Flux.update!(opt_state_discriminator, model.discriminator, grads_crit[1])

    return loss_c
end

function _train_vae!(model::GenerativeAdversarialNetwork, real_data, β, opt_state_vae_model)
    discriminator = model.discriminator

    loss_vae, grads_model = Flux.withgradient(AutoEnzyme(), model.vae_model) do vae_model
        fake_data = _decode(vae_model, _gan_make_latent_variables(model, real_data))
        recon_data = _decode(vae_model, _encode(vae_model, real_data))

        disc_fake_data = discriminator(fake_data)
        disc_recon_data = discriminator(recon_data)

        vae_elbo = _vae_elbo(vae_model, β, real_data)
        wgan_loss = - 0.5 .* (mean(disc_fake_data) + mean(disc_recon_data))
        return vae_elbo + wgan_loss
    end
    
    Flux.update!(opt_state_vae_model, model.vae_model, grads_model[1])

    return loss_vae
end

function load_model!(dst::GenerativeAdversarialNetwork, src::GenerativeAdversarialNetwork)
    Flux.loadmodel!(dst.discriminator, src.discriminator)
    load_model!(dst.vae_model, src.vae_model)
end

function _train!(protocol::GenerativeModelProtocol, model::GenerativeAdversarialNetwork, β::Float64; 
    print_log::Bool, critic_subepochs::Int, grad_penalty::Bool, λ::Float64, a::Float64,
    weight_clipping::Bool, clip_value::Float64)

    dev = protocol.device
    model_device = model |> dev
    training_data_device = protocol.training_data |> dev
    batchsize_device = protocol.batchsize
    shuffle_device = protocol.shuffle

    opt_discriminator = deepcopy(protocol.optimiser)
    opt_vae_model = deepcopy(protocol.optimiser)

    loader = load_data(training_data_device, batchsize_device, shuffle_device)

    opt_state_vae_model = Flux.setup(opt_vae_model, model_device.vae_model)
    opt_state_discriminator = Flux.setup(opt_discriminator, model_device.discriminator)

    for epoch in 1:protocol.epochs
        if epoch % 100 == 0 && protocol.shuffle
            loader = load_data(training_data_device, batchsize_device, shuffle_device)
        end

        running_loss_critic = 0.0
        running_loss_vae = 0.0

        for real_data in loader
            loss_critic = 0.0
            for _ in 1:maximum((1,critic_subepochs))
                loss_critic = _train_discriminator!(model_device, real_data, opt_state_discriminator; 
                    grad_penalty     = grad_penalty,
                    λ                = λ, 
                    a                = a,
                    weight_clipping  = weight_clipping,
                    clip_value       = clip_value
                )
            end

            loss_vae = _train_vae!(model_device, real_data, β, opt_state_vae_model)
            
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

    load_model!(protocol.model, model_device)

    return protocol._log
end

function _train!(protocol::GenerativeModelProtocol, model::GenerativeAdversarialNetwork; 
    print_log::Bool = true, critic_subepochs::Int = 5, β::Union{Float64, Vector{Float64}} = 1.0,
    grad_penalty::Bool = false, λ::Float64 = 10.0, a::Float64 = 1.0,
    weight_clipping::Bool = false, clip_value::Float64 = 1.0)

    for i in eachindex(β)
        _train!(protocol, model, β[i]; 
            print_log = print_log, critic_subepochs = critic_subepochs,
            grad_penalty = grad_penalty, λ = λ, a = a, 
            weight_clipping = weight_clipping, clip_value = clip_value
        )
    end

    return protocol._log
end

function _eval(model::GenerativeAdversarialNetwork, n_samples::Int)
    _eval(model.vae_model, n_samples)
end

function _eval(model::GenerativeAdversarialNetwork)
    return _eval(model,1)[:]
end

function _categorize(model::GenerativeAdversarialNetwork, x::Matrix)::Matrix
    return model.discriminator_network(x)
end

function _categorize(model::GenerativeAdversarialNetwork, x::Vector)::Vector
    return _categorize(model, reshape(x, :, 1))[:]
end

function _encode(model::GenerativeAdversarialNetwork, x::Matrix)::Matrix
    return _encode(model.vae_model, x)
end

function _encode(model::GenerativeAdversarialNetwork, x::Vector)::Vector
    return _encode(model, reshape(x, :, 1))[:]
end

function _decode(model::GenerativeAdversarialNetwork, z::Matrix)::Matrix
    return _decode(model.vae_model, z)
end

function _decode(model::GenerativeAdversarialNetwork, z::Vector)::Vector
    return _decode(model, reshape(z, :, 1))[:]
end

