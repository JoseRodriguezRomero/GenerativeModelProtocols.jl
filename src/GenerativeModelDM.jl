function default_denoiser_network(num_inputs::Int, hidden_layer_size::Int = 32)
    return Chain(
        Dense(num_inputs => hidden_layer_size, relu), 
        Dense(hidden_layer_size => hidden_layer_size, relu),
        Dense(hidden_layer_size => hidden_layer_size, relu),
        Dense(hidden_layer_size => hidden_layer_size, relu),
        Dense(hidden_layer_size => hidden_layer_size, relu),
        Dense(hidden_layer_size => num_inputs)
    ) |> f64
end

"""
$TYPEDEF

A structure containing the general parameters needed to evaluate and train 
a Diffusion Model (DM). Once trained, its denoiser model can be used as a 
generative model.

$TYPEDFIELDS
"""
@kwdef struct DiffusionModel <: AbstractGenerativeModel
    """Number of diffusion steps (number of steps in the Markov chain)."""
    T::Int
    """Float64(1.0) .- β"""
    α::Vector{Float64}
    """cumprod(α)"""
    ᾱ::Vector{Float64}
    """Variance of the Gaussian noise added in each diffusion step."""
    β::Vector{Float64}
    """Neural network parametrizing the denoising model, that is, a model that probabilisitically undoes the Gaussian noise."""
    denoiser_model::Chain
    """Number of training epochs for the VAE."""
end

function DiffusionModel(T::Int, β::Vector{Float64}, denoiser_model::Chain)
    α = Float64(1.0) .- β
    ᾱ = cumprod(α)
    return DiffusionModel(
        T = T,
        α = α,
        ᾱ = ᾱ,
        β = β,
        denoiser_model = denoiser_model
    )
end

function _generative_model(::DiffusionModel)::GenerativeModel
    return diffusion_model
end

function DiffusionModel(T::Int, β_start::Float64, β_end::Float64, denoiser_model::Chain)
    return DiffusionModel(T, Float64.(collect(range(β_start, β_end, length=T))), denoiser_model)
end

function DiffusionModel(num_inputs::Int, T::Int = 5, β_start::Float64=1.0E-4, β_end::Float64=0.02)
    return DiffusionModel(T, β_start, β_end, default_denoiser_network(num_inputs))
end

function load_diffusion_model_parameters(saved_model::Any;
    main_group_name::String = @default_main_group_name,
    generative_model_group_name::String = @default_generative_model_group_name)
    throw(ArgumentError("Types $(typeof(saved_model)) does not implement the required `load_diffusion_model_parameters` interface."))
end

function DiffusionModel(saved_model::Any)
    diffusion_model_parameters = load_diffusion_model_parameters(saved_model)

    return DiffusionModel(
        diffusion_model_parameters.T,
        diffusion_model_parameters.β,
        Chain([layer_parameters(layer) for layer in diffusion_model_parameters.denoiser_model.layers]...)
    )
end

function forward_diffusion(model::DiffusionModel, x₀::AbstractMatrix, t::Vector{Int})
    ϵ = Flux.randn_like(x₀, size(x₀))
    ᾱₜ = reshape(model.ᾱ[t], 1, :)
    xₜ = sqrt.(ᾱₜ) .* x₀ + sqrt.(Float64(1.0) .- ᾱₜ) .* ϵ
    return xₜ, ϵ
end

function _train!(protocol::GenerativeModelProtocol, model::DiffusionModel; print_log::Bool = true)
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
    loader_length = length(loader)
    loader_length_device = loader_length |> protocol.device

    if print_log; println("Training Diffusion Model...") end
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

        for x₀ in loader
            b_size = size(x₀, 2)

            t_raw = rand(1:model.T, b_size)
            xₜ, ϵ_true = forward_diffusion(model, x₀, t_raw)

            loss, grads = Flux.withgradient(model_train_device) do m
                ϵ_pred = m.denoiser_model(xₜ)
                Flux.Losses.mse(ϵ_pred, ϵ_true) / loader_length_device
            end

            raw_gradient_arrays = Optimisers.trainables(grads[1])
            batch_grad_norm = sqrt(sum(sum(abs2, g) for g in raw_gradient_arrays if g isa AbstractArray))

            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss
            total_grad_norm += batch_grad_norm
        end

        protocol._log.loss[epoch] = epoch_loss / length(loader)
        protocol._log.loss_grad_norm[epoch] = total_grad_norm / length(loader)

        if (epoch % 5 == 0 || epoch == 1) && print_log
            println("Epoch: $epoch | Mean MSE Loss: $(protocol._log.loss[epoch])")
        end
    end
    if print_log; println("Training complete!") end

    Flux.loadmodel!(model.denoiser_model, model_train_device.denoiser_model)

    return protocol._log
end

function _eval(protocol::GenerativeModelProtocol, model::DiffusionModel, n_samples::Int)
    x = randn(Float64,n_samples,size(protocol.training_data,1))
    x = collect(transpose(x))

    for t in model.T:-1:1
        ϵ_pred = model.denoiser_model(x)
        βₜ = model.β[t]
        αₜ = model.α[t]
        ᾱₜ = model.ᾱ[t]

        coeff = βₜ / sqrt(1.0 - ᾱₜ)
        σₜ = sqrt(βₜ)
        μₜ = (1.0 / sqrt(αₜ)) .* (x .- coeff .* ϵ_pred)

        if t > 1
            z = randn(Float64, size(x))
            x = μₜ .+ σₜ .* z
        else
            x = μₜ
        end
    end

    return collect(transpose(x))
end

struct DiffusionModelParameters
    T::Int
    β::Vector{Float64}
    denoiser_model::ChainParameters
end

function diffusion_model_parameters(model::DiffusionModel)
    return DiffusionModelParameters(
        model.T,
        model.β,
        chain_parameters(model.denoiser_model)
    )
end

