macro _dm_default_activation_function()
    return swish
end

function default_denoiser_network(
    num_inputs::Int, 
    T::Int, 
    precision::Type{FP}, 
    hidden_layer_size::Int = 32, 
    activation_function = @_dm_default_activation_function
    )::TabularDenoiser{FP} where {FP<:AbstractFloat}

    return TabularDenoiser(num_inputs; 
        T                   = T,
        hidden_layer_size   = hidden_layer_size,
        activation_function = activation_function,
        max_period          = precision(@tabular_denoiser_default_max_period)
    )
end

function compatible_dm_model(β::Tuple{Vararg{F}}, denoiser_model::TabularDenoiser{F}) where {F<:AbstractFloat}
    if length(β) != denoiser_model.T
        return false
    end

    return true
end

@compat public DiffusionModel

"""
$TYPEDEF

A structure containing the general parameters needed to evaluate and train 
a Diffusion Model (DM). Once trained, its denoiser model can be used as a 
generative model.

$TYPEDFIELDS
"""
@kwdef struct DiffusionModel{F<:AbstractFloat} <: AbstractGenerativeModel
    """Variance of the Gaussian noise added in each diffusion step."""
    β::Tuple{Vararg{F}}
    """Neural network parametrizing the denoising model, that is, a model that probabilisitically undoes the Gaussian noise."""
    denoiser_model::TabularDenoiser{F}
    """Number of training epochs for the VAE."""

    function DiffusionModel(β::Tuple{Vararg{F}}, denoiser_model::TabularDenoiser{F}) where {F<:AbstractFloat}
        if !compatible_dm_model(β, denoiser_model)
            @error "Incompatible DiffusionModel architecture!"
            throw(MethodError(DiffusionModel, (β, denoiser_model)))
        end

        return new{F}(β, denoiser_model)
    end
end

function _base_cast_diffusion_model(model::DiffusionModel, FP::Function)
    new_β = map(eltype(FP([1.0])), model.β) 
    new_denoiser_model = FP(model.denoiser_model)
    return DiffusionModel(new_β, new_denoiser_model)
end

function Flux.f16(model::DiffusionModel)
    return _base_cast_diffusion_model(model, Flux.f16)
end

function Flux.f32(model::DiffusionModel)
    return _base_cast_diffusion_model(model, Flux.f32)
end

function Flux.f64(model::DiffusionModel)
    return _base_cast_diffusion_model(model, Flux.f64)
end

"""
    GenerativeModelProtocols.DiffusionModel(num_inputs::Int, β::Tuple{Vararg{<:AbstractFloat}})

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

Sets the values of `α` and `ᾱ` automatically for the user. A default 
`denoiser_model` is created based on `num_inputs`.
"""
function DiffusionModel(num_inputs::Int, β::Tuple{Vararg{<:AbstractFloat}})
    T = length(β)
    denoiser_model = default_denoiser_network(num_inputs, T, eltype(β))
    return DiffusionModel(β, denoiser_model)
end

function _generative_model(::DiffusionModel)::GenerativeModel
    return diffusion_model
end

"""
    GenerativeModelProtocols.DiffusionModel(T::Int, β_start::FP, β_end::FP, denoiser_model::TabularDenoiser{FP}) where {FP<:AbstractFloat}

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

Sets the values of `α` and `ᾱ` automatically for the user, using user specified 
`denoiser_model`. This constructor initializes `β` with 
`collect(range(β_start, β_end, length=T))`.
"""
function DiffusionModel(T::Int, β_start::FP, β_end::FP, denoiser_model::TabularDenoiser{FP}) where {FP<:AbstractFloat}
    return DiffusionModel(Tuple(collect(range(β_start, β_end, length=T))), denoiser_model)
end

"""
    GenerativeModelProtocols.DiffusionModel(num_inputs::Int, T::Int = 5, β_start::FP = 1.0E-4, β_end::FP = 0.02) where {FP<:AbstractFloat}

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

Sets the values of `α` and `ᾱ` automatically for the user. A default 
`denoiser_model` is created based on `num_inputs`. This constructor initializes 
`β` with `collect(range(β_start, β_end, length=T))`.
"""
function DiffusionModel(num_inputs::Int, T::Int = 5, β_start::FP = 1.0E-4, β_end::FP = 0.02) where {FP<:AbstractFloat}
    denoiser_model = default_denoiser_network(num_inputs, T, eltype(β_start))
    return DiffusionModel(T, β_start, β_end, denoiser_model)
end

function load_diffusion_model_parameters end

macro load_diffusion_model_parameters(saved_model, main_group_name, generative_model_group_name, tabular_denoiser_group_name)
    return :(load_diffusion_model_parameters($(esc(saved_model)); 
        $(main_group_name = esc(main_group_name)), 
        $(generative_model_group_name = esc(generative_model_group_name)),
        $(tabular_denoiser_group_name = esc(tabular_denoiser_group_name))
    ))
end

function DiffusionModel(saved_model::String; 
    main_group_name::String = @default_main_group_name,
    generative_model_group_name::String = @default_generative_model_group_name,
    tabular_denoiser_group_name::String = @default_tabular_denoiser_group_name)

    return @load_diffusion_model_parameters(saved_model, main_group_name, generative_model_group_name, tabular_denoiser_group_name)
end

function Base.display(model::DiffusionModel)
    println("$(summary(model)):")
    println("β              = $(summary(model.β))")
    println("denoiser_model = $(summary(model.denoiser_model))")
end

function _input_size(model::DiffusionModel)::Int
    return _input_size(model.denoiser_model.input_projection)
end

function _latent_size(model::DiffusionModel)::Int
    return _input_size(model)
end 

function forward_diffusion(ᾱ::Tuple{Vararg{<:AbstractFloat}}, x₀::Matrix, t::Vector{Int})
    FP = eltype(ᾱ)
    ϵ = Flux.randn_like(x₀, FP, size(x₀))
    
    ᾱₜ = reshape(view(collect(ᾱ), t), 1, :)
    xₜ = sqrt.(ᾱₜ) .* x₀ + sqrt.(FP(1.0) .- ᾱₜ) .* ϵ
    
    return xₜ, ϵ
end

function forward_diffusion(model::DiffusionModel, x₀::Matrix, t::Vector{Int})
    FP = eltype(x₀)

    β = model.β
    α = FP(1.0) .- β
    ᾱ = cumprod(α)

    return forward_diffusion(ᾱ, x₀, t)
end

function forward_diffusion(model::DiffusionModel, x₀::Matrix, t::Int)
    return forward_diffusion(model, x₀, [t])
end

function _train!(protocol::GenerativeModelProtocol, model::DiffusionModel; print_log::Bool = true)
    model_train_device = model.denoiser_model |> protocol.device
    opt_state = Flux.setup(protocol.optimiser, model_train_device)
    
    batchsize_device = protocol.batchsize |> protocol.device
    training_data_device = protocol.training_data |> protocol.device

    loader = Flux.DataLoader(
        training_data_device, 
        batchsize = batchsize_device, 
        shuffle = false,
        parallel = true
    )
    loader_length = length(loader)
    loader_length_device = loader_length |> protocol.device

    FP = eltype(protocol.precision([1.0]))

    β = model.β
    α = FP(1.0) .- β
    ᾱ = cumprod(α)
    ᾱ_device = ᾱ |> protocol.device

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
            t_raw = rand(1:model.denoiser_model.T, size(x₀, 2))
            xₜ, ϵ_true = forward_diffusion(ᾱ_device, x₀, t_raw)

            loss, (grads,) = Flux.withgradient(AutoEnzyme(), model_train_device) do denoiser_model
                ϵ_pred = denoiser_model(xₜ, t_raw)
                return Flux.Losses.mse(ϵ_pred, ϵ_true)
            end

            raw_gradient_arrays = Optimisers.trainables(grads)
            batch_grad_norm = sqrt(sum(sum(abs2, g) for g in raw_gradient_arrays if g isa AbstractArray))

            Flux.update!(opt_state, model_train_device, grads)
            
            epoch_loss += loss / loader_length_device
            total_grad_norm += batch_grad_norm / loader_length_device
        end

        protocol._log.loss[epoch] = epoch_loss
        protocol._log.loss_grad_norm[epoch] = total_grad_norm

        if (epoch % 5 == 0 || epoch == 1) && print_log
            @printf("Epoch: %8d | Mean MSE Loss: %-15.8f | Grad Norm: %-10.8f\n", epoch, protocol._log.loss[epoch], protocol._log.loss_grad_norm[epoch])
        end
    end
    if print_log; println("Training complete!") end

    Flux.loadmodel!(model.denoiser_model, model_train_device)

    return protocol._log
end

function _encode(model::DiffusionModel, x::Matrix)::Matrix
    z, _ = forward_diffusion(model, x, model.denoiser_model.T)
    return z
end

function _encode(model::DiffusionModel, x::Vector)::Vector
    return _encode(model, reshape(x, :, 1))[:]
end

function _decode(model::DiffusionModel, z::Matrix)::Matrix
    FP = eltype(z)

    β = model.β
    α = FP(1.0) .- β
    ᾱ = cumprod(α)

    x = z
    for t in model.denoiser_model.T:-1:1
        t_batch = fill(t, size(z, 2))
        ϵ_pred = model.denoiser_model(x, t_batch)
        x_mean = (x .- (β[t] / sqrt(FP(1.0) - ᾱ[t])) .* ϵ_pred) ./ sqrt(α[t]) 
        
        if t > 1
            σ_t = sqrt(β[t] * (FP(1.0) - ᾱ[t-1]) / (FP(1.0) - ᾱ[t]))
            x = x_mean .+ σ_t .* Flux.randn_like(x, size(x))
        else
            x = x_mean
        end
    end
    
    return x
end

function _decode(model::DiffusionModel, z::Vector)::Vector
    return _decode(model, reshape(z, :, 1))[:]
end

function _eval(model::DiffusionModel, n_samples::Int)
    num_features = _output_size(model.denoiser_model.output_projection)
    z = Flux.randn_like([model.β[1]], (num_features, n_samples))
    
    return _decode(model, z)
end

function _eval(model::DiffusionModel)
    return _eval(model, 1)[:]
end

