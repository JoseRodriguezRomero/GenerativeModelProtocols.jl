macro _gan_default_activation_function()
    return leakyrelu
end

function _gan_default_generator_network(latent_size::Int, input_size::Int; hidden_layer_size::Int = 32, activation_function::Function = @_gan_default_activation_function)
    return Chain(
        Dense(latent_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => input_size)
    ) |> f64
end

function _gan_default_discriminator_network(input_size::Int; hidden_layer_size::Int = 32, activation_function::Function = @_gan_default_activation_function)
    return Chain(
        Dense(input_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 1)
    ) |> f64
end

function _gan_default_encoder_network(input_size::Int, latent_size::Int; hidden_layer_size::Int = 32, activation_function::Function = @_gan_default_activation_function)
    return Chain(
        Dense(input_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => 2 * latent_size)
    ) |> f64
end

function compatible_gan_model(generator_network::Chain, discriminator_network::Chain, encoder_network::Chain)::Bool
    if output_size(generator_network) != input_size(discriminator_network)
        return false
    end

    if (2 * input_size(generator_network)) != output_size(encoder_network)
        return false
    end

    if input_size(encoder_network) != output_size(generator_network)
        return false
    end

    if output_size(discriminator_network) != 1
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
    """Neural network for making synthetic data."""
    generator_network::Chain
    """Neural network discriminating between real and synthetic data."""
    discriminator_network::Chain
    """Neural network for encoding data into a latent space."""
    encoder_network::Chain

    function GenerativeAdversarialNetwork(generator_network::Chain, discriminator_network::Chain, encoder_network::Chain)
        if !compatible_gan_model(generator_network, discriminator_network, encoder_network)
            @error "Incompatible GenerativeAdversarialNetwork architecture!"
            throw(MethodError(GenerativeAdversarialNetwork, (generator_network, discriminator_network, encoder_network)))
        end

        return new(generator_network, discriminator_network, encoder_network)
    end
end

"""
    GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size::Int, latent_size::Int)

Convenience constructor that creates a 
`GenerativeModelProtocols.GenerativeAdversarialNetwork` using default generator
and discriminator networks.
"""
function GenerativeAdversarialNetwork(input_size::Int, latent_size::Int)
    return GenerativeAdversarialNetwork(;
        generator_network     = _gan_default_generator_network(latent_size, input_size),
        discriminator_network = _gan_default_discriminator_network(input_size),
        encoder_network       = _gan_default_encoder_network(input_size, latent_size)
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

    println("generator_network: ")
    _print_chains(model.generator_network, print_padding)

    println("")

    println("discriminator_network: ")
    _print_chains(model.discriminator_network, print_padding)
end

function _generative_model(::GenerativeAdversarialNetwork)::GenerativeModel
    return generative_adversarial_network
end

function _gan_encode(encoder_network::Chain, x::Matrix)
    enc_out = encoder_network(x)
    μ = enc_out[1:2:end, :]
    log_σ² = enc_out[2:2:end, :]
    ϵ = Flux.randn_like(μ, size(x))

    return μ .+ exp.(0.5 .* log_σ²) .* ϵ
end

function _train_discriminator!(model::GenerativeAdversarialNetwork, real_data, opt_state_crit; 
    grad_penalty::Bool, grad_finite_diff::Bool, ϵ, λ::Float64, a::Float64, 
    weight_clipping::Bool, clip_value::Float64)

    gen = model.generator_network
    crit = model.discriminator_network
    enc = model.encoder_network

    fake_data = Zygote.dropgrad(gen(_gan_encode(enc, real_data)))
    interpolates = ϵ .* real_data .+ (1.0 .- ϵ) .* fake_data

    loss_c, grads_crit = Flux.withgradient(crit) do c_net
        crit_real = c_net(real_data)
        crit_fake = c_net(fake_data)
        w_loss = Flux.mean(crit_fake) - Flux.mean(crit_real)

        if grad_penalty
            if grad_finite_diff
                h = 1.0E-8
                gp = 0.0
                N_features = size(real_data, 1)

                for i in 1:N_features
                    mask = (1:N_features .== i)
                    interpolates_perturbed = interpolates .+ (mask .* h)

                    delta_output = c_net(interpolates_perturbed) .- c_net(interpolates)
                    approx_grad = delta_output ./ h

                    gp += Flux.mean(max.(0.0, abs.(approx_grad) .- a).^2)
                end

                return w_loss + λ * gp
            else
                (grads_interp,) = Zygote.gradient(interpolates) do x
                    sum(c_net(x))
                end

                grad_norms = sqrt.(sum(abs2, grads_interp, dims=1) .+ 1.0E-8)
                gp = Flux.mean(abs2.(grad_norms .- a))
                return w_loss + λ * gp
            end
        else
            return w_loss
        end
    end
    
    Flux.update!(opt_state_crit, crit, grads_crit[1])

    if weight_clipping
        foreach(Flux.trainable(crit)) do layer_params
            foreach(layer_params) do p
                if p isa AbstractArray
                    p .= clamp.(p, -clip_value, clip_value)
                end
            end
        end
    end

    return loss_c
end

function _train_generator!(model::GenerativeAdversarialNetwork, real_data, opt_state_gen)
    gen = model.generator_network
    crit = model.discriminator_network
    enc = model.encoder_network

    loss_g, grads_gen = Flux.withgradient(gen) do g_net
        noise = _gan_encode(enc, real_data)
        fake_data = g_net(noise)
        crit_fake = crit(fake_data)

        return -Flux.mean(crit_fake)
    end
    
    Flux.update!(opt_state_gen, gen, grads_gen[1])
    return loss_g
end

function _train_encoder!(model::GenerativeAdversarialNetwork, real_data, λ, opt_state_enc)
    enc = model.encoder_network
    gen = model.generator_network

    loss_e, grads_enc = Flux.withgradient(enc) do e_net
        enc_out = e_net(real_data)
        μ = enc_out[1:2:end, :]
        log_σ² = enc_out[2:2:end, :]

        z = _gan_encode(e_net, real_data)
        x̂ = gen(z)

        recon_loss = 0.5 * Flux.mean(sum((real_data .- x̂) .^ 2, dims = 1))
        kl_loss = 0.5 * λ * Flux.mean(sum(μ.^2 .+ exp.(log_σ²) .- log_σ² .- 1.0, dims = 1))

        return recon_loss + kl_loss
    end

    Flux.update!(opt_state_enc, enc, grads_enc[1])
    return loss_e
end

function _train!(protocol::GenerativeModelProtocol, model::GenerativeAdversarialNetwork; 
    print_log::Bool = true, λ::Float64 = 1.0,
    grad_penalty::Bool = false, grad_finite_diff::Bool = true, λ_grad::Float64 = 10.0, a::Float64 = 1.0,
    weight_clipping::Bool = false, clip_value::Float64 = 1.0,)

    dev = protocol.device
    gen = model.generator_network |> dev
    crit = model.discriminator_network |> dev
    enc = model.encoder_network |> dev

    training_data_device = protocol.training_data |> dev
    batchsize_device = protocol.batchsize
    shuffle_device = protocol.shuffle

    opt_gen = deepcopy(protocol.optimiser)
    opt_crit = deepcopy(protocol.optimiser)
    opt_enc = deepcopy(protocol.optimiser)

    loader = load_data(training_data_device, batchsize_device, shuffle_device)

    opt_state_gen = Flux.setup(opt_gen, gen)
    opt_state_crit = Flux.setup(opt_crit, crit)
    opt_state_enc = Flux.setup(opt_enc, enc)

    for epoch in 1:protocol.epochs
        if epoch % 100 == 0 && protocol.shuffle
            loader = load_data(training_data_device, batchsize_device, shuffle_device)
        end

        running_loss_c = 0.0
        running_loss_g = 0.0
        running_loss_e = 0.0
        running_w_dist = 0.0
        steps = 0

        for real_data in loader
            steps += 1

            dims = (ones(Int, ndims(real_data) - 1)..., size(real_data, ndims(real_data)))
            ϵ = Flux.rand_like(real_data, dims)

            loss_c = _train_discriminator!(model, real_data, opt_state_crit; 
                grad_penalty     = grad_penalty,
                grad_finite_diff = grad_finite_diff,
                ϵ                = ϵ, 
                λ                = λ_grad, 
                a                = a,
                weight_clipping  = weight_clipping,
                clip_value       = clip_value
            )

            loss_g = _train_generator!(model, real_data, opt_state_gen)
            w_dist = - (loss_g + loss_c)

            loss_e = _train_encoder!(model, real_data, λ, opt_state_enc)
            
            running_loss_c += loss_c
            running_loss_g += loss_g
            running_loss_e += loss_e
            running_w_dist += w_dist
        end
        
        if print_log && steps > 0
            avg_loss_c = running_loss_c / steps
            avg_loss_g = running_loss_g / steps
            avg_loss_e = running_loss_e / steps
            avg_w_dist = running_w_dist / steps
        
            @printf("Epoch %5d", epoch)
            @printf(" | Avg. Crit. Loss: %15.6E", avg_loss_c)
            @printf(" | Avg. Gen. Loss: %15.6E", avg_loss_g)
            @printf(" | Avg. Enc. loss: %15.6E", avg_loss_e)
            @printf(" | Avg. Wasserstein dist: %15.6E \n", avg_w_dist)
        end
    end


    Flux.loadmodel!(model.generator_network, gen)
    Flux.loadmodel!(model.discriminator_network, crit)

    return protocol._log
end

function _eval(model::GenerativeAdversarialNetwork, n_samples::Int)
    latent_size = input_size(model.generator_network)
    noise = randn(Float64, latent_size, n_samples)
    synthetic_data = model.generator_network(noise)
    return synthetic_data
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

