macro _dm_default_activation_function()
    return swish
end

function default_denoiser_network(
    num_inputs::Int, 
    T::Int, 
    precision::Type{FP}, 
    hidden_layer_size::Int = 32, 
    activation_function = @_dm_default_activation_function
    )::TabularDenoiser where {FP<:AbstractFloat}

    return TabularDenoiser(num_inputs; 
        T                   = T,
        hidden_layer_size   = hidden_layer_size,
        activation_function = activation_function,
        max_period          = precision(@tabular_denoiser_default_max_period)
    )
end

function compatible_dm_model(β::Tuple{Vararg{<:AbstractFloat}}, denoiser_model::TabularDenoiser{LayerNames}) where {LayerNames}
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
@kwdef struct DiffusionModel{LayerNames} <: AbstractGenerativeModel
    """Variance of the Gaussian noise added in each diffusion step."""
    β::Tuple{Vararg{<:AbstractFloat}}
    """Neural network parametrizing the denoising model, that is, a model that probabilisitically undoes the Gaussian noise."""
    denoiser_model::TabularDenoiser{LayerNames}
    """Number of training epochs for the VAE."""

    function DiffusionModel(β::Tuple{Vararg{<:AbstractFloat}}, denoiser_model::TabularDenoiser{LayerNames}) where LayerNames
        if !compatible_dm_model(β, denoiser_model)
            @error "Incompatible DiffusionModel architecture!"
            throw(MethodError(DiffusionModel, (β, denoiser_model)))
        end

        return new{LayerNames}(β, denoiser_model)
    end
end

DiffusionModel{LayerNames}(args...; kwargs...) where LayerNames = DiffusionModel(args...; kwargs...)

"""
    GenerativeModelProtocols.DiffusionModel(num_inputs::Int, β::Tuple{Vararg{AbstractFloat}}})

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

A default `denoiser_model` is created based on `num_inputs`.
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
    GenerativeModelProtocols.DiffusionModel(T::Int, β_start::FP, β_end::FP, denoiser_model::TabularDenoiser{LayerNames}) where {LayerNames, FP<:AbstractFloat}

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

This constructor initializes `β` with 
`collect(range(β_start, β_end, length=T))`.
"""
function DiffusionModel(T::Int, β_start::FP, β_end::FP, denoiser_model::TabularDenoiser{LayerNames}) where {LayerNames, FP<:AbstractFloat}
    return DiffusionModel(Tuple(collect(range(β_start, β_end, length=T))), denoiser_model)
end

"""
    GenerativeModelProtocols.DiffusionModel(num_inputs::Int, T::Int = 5, β_start::FP = 1.0E-4, β_end::FP = 0.02) where {FP<:AbstractFloat}

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

A default `denoiser_model` is created based on `num_inputs`. This constructor 
initializes `β` with `collect(range(β_start, β_end, length=T))`.
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
    println("$(Base.typename(typeof(model)).wrapper):")
    println("β              = $(summary(model.β))")
    println("denoiser_model = $(Base.typename(typeof(model.denoiser_model)).wrapper)")
end

function _input_size(model::DiffusionModel)::Int
    return _input_size(model.denoiser_model.input_projection)
end

function _latent_size(model::DiffusionModel)::Int
    return _input_size(model)
end 

function forward_diffusion(ᾱ::Tuple{Vararg{AbstractFloat}}, x₀, t)
    FP = eltype(ᾱ)
    ϵ = randn_like(x₀, FP, size(x₀))
    
    ᾱₜ = reshape(view(collect(ᾱ), t), 1, :)
    xₜ = sqrt.(ᾱₜ) .* x₀ + sqrt.(FP(1.0) .- ᾱₜ) .* ϵ
    
    return xₜ, ϵ
end

function forward_diffusion(model::DiffusionModel, x₀, t)
    FP = eltype(x₀)

    β = model.β
    α = FP(1.0) .- β
    ᾱ = cumprod(α)

    return forward_diffusion(ᾱ, x₀, t)
end

function forward_diffusion(model::DiffusionModel, x₀, t::Int)
    return forward_diffusion(model, x₀, [t])
end

function _train!(protocol::GenerativeModelProtocol, model::DiffusionModel; print_log::Bool = true)
    denoiser_model = model.denoiser_model
    training_data_device = protocol.training_data |> protocol.device

    loader = load_data(training_data_device, protocol.batchsize, protocol.shuffle)
    T = eltype(protocol.training_data)

    β = model.β
    α = T(1.0) .- β
    ᾱ = cumprod(α)
    ᾱ_device = ᾱ |> protocol.device

    protocol._log["Mean MSE"] = zeros(T, protocol.epochs)

    function _dm_train_step!(x, p_current, s_current, o_current)
        num_steps = denoiser_model.T
        batch_size = size(x, 2)

        t_uniform = rand_like(x, eltype(x), (batch_size,))
        t_raw = floor.(t_uniform .* num_steps) .+ 1
        t_raw_int = Int.(t_raw)
        xₜ, ϵ_true = forward_diffusion(ᾱ_device, x, t_raw_int)

        _objective = (denoiser_model, p, s) -> begin
            ϵ_pred, _ = _eval(denoiser_model, xₜ, t_raw, p, s)
            return mean(abs2, ϵ_pred .- ϵ_true)
        end

        loss_val = _objective(denoiser_model, p_current, s_current)
        loss_grads = Enzyme.make_zero(p_current)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(_objective),
            Enzyme.Active,
            Enzyme.Const(denoiser_model),
            Enzyme.Duplicated(p_current, loss_grads),
            Enzyme.Const(s_current)
        )

        o_updated, p_updated = Optimisers.update(o_current, p_current, loss_grads)

        return loss_val, p_updated, o_updated
    end

    ps = protocol.precision(model.denoiser_model._ps[]) |> protocol.device
    st = protocol.precision(model.denoiser_model._st[]) |> protocol.device

    opt_state = _initial_step(model.denoiser_model, ps, st, protocol.optimiser)
    _train_step!, opt_state = _train_step_device_dispatch(protocol.device, _dm_train_step!, loader, opt_state)

    if print_log; println("Training Diffusion Model...") end
    for epoch in 1:protocol.epochs
        epoch_loss = T(0.0)

        if epoch % 100 == 0 && protocol.shuffle
            loader = load_data(training_data_device, protocol.batchsize, protocol.shuffle)
        end

        for x₀ in loader
            loss, opt_state = _train_step!(x₀, opt_state)
            epoch_loss += loss
        end

        mean_loss = epoch_loss / length(loader)
        protocol._log["Mean MSE"][epoch] = mean_loss

        if print_log && (epoch % 5 == 0 || epoch == 1)
            @printf("Epoch: %8d | Mean MSE Loss: %-15.8f \n", epoch, mean_loss)
        end
    end
    if print_log; println("Training complete!") end

    model.denoiser_model._ps[] = opt_state.parameters
    model.denoiser_model._st[] = opt_state.states

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
        t_batch = FP.(fill(t, size(z, 2)))
        ϵ_pred = model.denoiser_model(x, t_batch)
        x_mean = (x .- (β[t] / sqrt(FP(1.0) - ᾱ[t])) .* ϵ_pred) ./ sqrt(α[t]) 
        
        if t > 1
            σ_t = sqrt(β[t] * (FP(1.0) - ᾱ[t-1]) / (FP(1.0) - ᾱ[t]))
            x = x_mean .+ σ_t .* randn_like(x, size(x))
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
    z = randn_like([model.β[1]], (num_features, n_samples))
    
    return _decode(model, z)
end

function _eval(model::DiffusionModel)
    return _eval(model, 1)[:]
end

