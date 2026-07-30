macro _dm_default_activation_function()
    return relu
end

function default_denoiser_network(num_inputs::Int, T::Int, hidden_layer_size::Int = 32, activation_function = @_dm_default_activation_function)
    return TabularDenoiser(num_inputs; 
        T                   = T,
        hidden_layer_size   = hidden_layer_size,
        activation_function = activation_function
    ) |> f64
end

@compat public DiffusionModel

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
    denoiser_model::TabularDenoiser
    """Number of training epochs for the VAE."""
end

"""
    GenerativeModelProtocols.DiffusionModel(β::Vector{Float64}, denoiser_model::GenerativeModelProtocols.TabularDenoiser)

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

Sets the values of `α` and `ᾱ` automatically for the user, using user specified 
`denoiser_model`. The user is responsible to ensuring that `denoiser_model` is
compatible with `β`.
"""
function DiffusionModel(β::Vector{Float64}, denoiser_model::TabularDenoiser)
    α = Float64(1.0) .- β
    ᾱ = cumprod(α)
    return DiffusionModel(
        T = length(β),
        α = α,
        ᾱ = ᾱ,
        β = β,
        denoiser_model = denoiser_model
    )
end

"""
    GenerativeModelProtocols.DiffusionModel(num_inputs::Int, T::Int, β::Vector{Float64})

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

Sets the values of `α` and `ᾱ` automatically for the user. A default 
`denoiser_model` is created based on `num_inputs`.
"""
function DiffusionModel(num_inputs::Int, β::Vector{Float64})
    return DiffusionModel(β = β, denoiser_model = default_denoiser_network(num_inputs, T))
end

function _generative_model(::DiffusionModel)::GenerativeModel
    return diffusion_model
end

"""
    GenerativeModelProtocols.DiffusionModel(T::Int, β_start::Float64, β_end::Float64, denoiser_model::GenerativeModelProtocols.TabularDenoiser)

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

Sets the values of `α` and `ᾱ` automatically for the user, using user specified 
`denoiser_model`. This constructor initializes `β` with 
`collect(range(β_start, β_end, length=T))`.
"""
function DiffusionModel(T::Int, β_start::Float64, β_end::Float64, denoiser_model::TabularDenoiser)
    return DiffusionModel(Float64.(collect(range(β_start, β_end, length=T))), denoiser_model)
end

"""
    GenerativeModelProtocols.DiffusionModel(num_inputs::Int, T::Int = 5, β_start::Float64=1.0E-4, β_end::Float64=0.02)

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

Sets the values of `α` and `ᾱ` automatically for the user. A default 
`denoiser_model` is created based on `num_inputs`. This constructor initializes 
`β` with `collect(range(β_start, β_end, length=T))`.
"""
function DiffusionModel(num_inputs::Int, T::Int = 5, β_start::Float64=1.0E-4, β_end::Float64=0.02)
    return DiffusionModel(T, β_start, β_end, default_denoiser_network(num_inputs, T))
end

function load_diffusion_model_parameters(saved_model::Any;
    main_group_name::String = @default_main_group_name,
    generative_model_group_name::String = @default_generative_model_group_name)
    throw(ArgumentError("Types $(typeof(saved_model)) does not implement the required `load_diffusion_model_parameters` interface."))
end

function DiffusionModel(saved_model::Any; 
    main_group_name::String = @default_main_group_name,
    generative_model_group_name::String = @default_generative_model_group_name)

    diffusion_model_parameters = load_diffusion_model_parameters(saved_model;
        main_group_name = main_group_name,
        generative_model_group_name = generative_model_group_name
    )

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
        epoch_loss = 0.0
        total_grad_norm = 0.0

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

            t_raw = rand(1:model.T, b_size) |> protocol.device
            xₜ, ϵ_true = forward_diffusion(model_train_device, x₀, t_raw)

            loss, grads = Flux.withgradient(model_train_device) do m
                ϵ_pred = m.denoiser_model(xₜ, t_raw)
                return Flux.Losses.mse(ϵ_pred, ϵ_true) / loader_length_device
            end

            g_tree = grads[1] 
            raw_gradient_arrays = Optimisers.trainables(g_tree)
            batch_grad_norm = sqrt(sum(sum(abs2, g) for g in raw_gradient_arrays if g isa AbstractArray))

            Flux.update!(opt_state, model_train_device, g_tree)
            
            epoch_loss += loss
            total_grad_norm += batch_grad_norm
        end

        protocol._log.loss[epoch] = epoch_loss / length(loader)
        protocol._log.loss_grad_norm[epoch] = total_grad_norm / length(loader)

        if (epoch % 5 == 0 || epoch == 1) && print_log
            @printf("Epoch: %8d | Mean MSE Loss: %-15.6f | Grad Norm: %-10.4f\n", epoch, protocol._log.loss[epoch], protocol._log.loss_grad_norm[epoch])
        end
    end
    if print_log; println("Training complete!") end

    Flux.loadmodel!(model.denoiser_model, model_train_device.denoiser_model)

    return protocol._log
end

function _eval(protocol::GenerativeModelProtocol, model::DiffusionModel, n_samples::Int)
    model_device = model |> protocol.device
    num_features = size(model_device.denoiser_model.output_projection.weight, 1)
    
    x = randn(Float64, num_features, n_samples) |> protocol.device
    
    α = model_device.α
    ᾱ = model_device.ᾱ
    β = model_device.β
    
    for t in model_device.T:-1:1
        t_batch = fill(t, n_samples) |> protocol.device
        ϵ_pred = model_device.denoiser_model(x, t_batch)
        x_mean = (x .- (β[t] / sqrt(1.0 - ᾱ[t])) .* ϵ_pred) ./ sqrt(α[t]) 
        
        if t > 1
            σ_t = sqrt(β[t] * (1.0 - ᾱ[t-1]) / (1.0 - ᾱ[t]))
            z = randn(Float64, size(x)...) |> protocol.device
            x = x_mean .+ σ_t .* z
        else
            x = x_mean
        end
    end
    
    return x |> cpu
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

