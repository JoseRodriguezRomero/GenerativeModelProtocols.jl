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

function compatible_dm_model(T::Int, α::Tuple{Vararg{Float64}}, ᾱ::Tuple{Vararg{Float64}}, β::Tuple{Vararg{Float64}}, denoiser_model::TabularDenoiser)
    if T != denoiser_model.T
        println("D")
        return false
    end

    if length(α) != T
        println("A")
        return false
    end

    if length(ᾱ) != T
        println("B")
        return false
    end

    if length(β) != T
        println("C")
        return false
    end

    if α != (1.0 .- β)
        println("E")
        return false
    end

    if ᾱ != cumprod(α)
        println("F")
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
@kwdef struct DiffusionModel <: AbstractGenerativeModel
    """Number of diffusion steps (number of steps in the Markov chain)."""
    T::Int
    """1.0 .- β"""
    α::Tuple{Vararg{Float64}}
    """cumprod(α)"""
    ᾱ::Tuple{Vararg{Float64}}
    """Variance of the Gaussian noise added in each diffusion step."""
    β::Tuple{Vararg{Float64}}
    """Neural network parametrizing the denoising model, that is, a model that probabilisitically undoes the Gaussian noise."""
    denoiser_model::TabularDenoiser
    """Number of training epochs for the VAE."""

    function DiffusionModel(T::Int, α::Tuple{Vararg{Float64}}, ᾱ::Tuple{Vararg{Float64}}, β::Tuple{Vararg{Float64}}, denoiser_model::TabularDenoiser)
        if !compatible_dm_model(T, α, ᾱ, β, denoiser_model)
            @error "Incompatible DiffusionModel architecture!"
            throw(MethodError(DiffusionModel, (T, α, ᾱ, β, denoiser_model)))
        end

        return new(T, α, ᾱ, β, denoiser_model)
    end
end

"""
    GenerativeModelProtocols.DiffusionModel(β::Vector{Float64}, denoiser_model::GenerativeModelProtocols.TabularDenoiser)

Convenience constructor to create a `GenerativeModelProtocols.DiffusionModel`.

Sets the values of `α` and `ᾱ` automatically for the user, using user specified 
`denoiser_model`. The user is responsible to ensuring that `denoiser_model` is
compatible with `β`.
"""
function DiffusionModel(β::Tuple{Vararg{Float64}}, denoiser_model::TabularDenoiser)
    α = 1.0 .- β
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
function DiffusionModel(num_inputs::Int, β::Tuple{Vararg{Float64}})
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
    return DiffusionModel(Tuple(collect(range(β_start, β_end, length=T))), denoiser_model)
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
    println("T              = $(model.T)")
    println("α              = $(summary(model.α))")
    println("ᾱ              = $(summary(model.ᾱ))")
    println("β              = $(summary(model.β))")
    println("denoiser_model = $(summary(model.denoiser_model))")
end

function forward_diffusion(model::DiffusionModel, x₀::Matrix, t::Vector{Int})
    ϵ = Flux.randn_like(x₀, size(x₀))
    ᾱₜ = reshape(collect(model.ᾱ[t]), 1, :)
    xₜ = sqrt.(ᾱₜ) .* x₀ + sqrt.(Float64(1.0) .- ᾱₜ) .* ϵ
    return xₜ, ϵ
end

function forward_diffusion(model::DiffusionModel, x₀::Matrix, t::Int)
    return forward_diffusion(model, x₀, [t])
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

"""
    encode(model::GenerativeModelProtocols.DiffusionModel, x::Matrix) -> Matrix

Encodes the data space variable `x` into a latent space variable.
"""
function encode(model::DiffusionModel, x::Matrix)::Matrix
    z, _ = forward_diffusion(model, x, model.T)
    return z
end

"""
    encode(model::GenerativeModelProtocols.DiffusionModel, x::Vector) -> Vector

Encodes the data space variable `x` into a latent space variable.
"""
function encode(model::DiffusionModel, x::Vector)::Vector
    return encode(model, reshape(x, :, 1))[:]
end

"""
    decode(model::GenerativeModelProtocols.DiffusionModel, z::Matrix) -> Matrix

Decodes the latent space representations `z` back into the data space.
"""
function decode(model::DiffusionModel, z::Matrix)::Matrix
    α = model.α
    ᾱ = model.ᾱ
    β = model.β

    x = z
    for t in model.T:-1:1
        t_batch = fill(t, size(z, 2))
        ϵ_pred = model.denoiser_model(x, t_batch)
        x_mean = (x .- (β[t] / sqrt(1.0 - ᾱ[t])) .* ϵ_pred) ./ sqrt(α[t]) 
        
        if t > 1
            σ_t = sqrt(β[t] * (1.0 - ᾱ[t-1]) / (1.0 - ᾱ[t]))
            z = randn(Float64, size(x)...)
            x = x_mean .+ σ_t .* z
        else
            x = x_mean
        end
    end
    
    return x
end

"""
    decode(model::GenerativeModelProtocols.DiffusionModel, z::Vector) -> Vector

Decodes the latent space representations `z` back into the data space.
"""
function decode(model::DiffusionModel, z::Vector)::Vector
    return decode(model, reshape(z, :, 1))[:]
end


function (model::DiffusionModel)(n_samples::Int)
    num_features = size(model.denoiser_model.output_projection.weight, 1)
    z = randn(Float64, num_features, n_samples)
    
    return decode(model, z)
end

function (model::DiffusionModel)()
    return model(1)[:]
end

