module GenerativeModelProtocols

using Lux
using Dates
using Printf
using Enzyme
using Random
using MLUtils
using StatsBase
using Optimisers
using LinearAlgebra
using DocStringExtensions

using Compat: @compat

export GenerativeModelProtocol, train!, categorize, encode, decode, input_size, latent_size

abstract type AbstractGenerativeModel end
abstract type AbstractCategoricalGenerativeModel <: AbstractGenerativeModel end

function compatible_generative_protocol(
    training_data::Union{Matrix{<:AbstractFloat}, Nothing},
    var_training_data::Tuple{Vararg{AbstractFloat}}
    )

    ϵ = 1.0E-9
    if !isnothing(training_data)
        if maximum(abs.(mean(training_data, dims = 2))) > ϵ
            return false
        end

        if maximum(abs.(var(training_data, dims = 2) .- 1.0)) > ϵ
            return false
        end
    end

    if minimum(var_training_data) < ϵ
        println(minimum(var_training_data))
        return false
    end

    return true
end

macro default_main_group_name()
    return "generative_model_protocol"
end

macro default_metadata_group_name()
    return "metadata"
end

macro default_generative_model_group_name()
    return "generative_model"
end

_cast_to_precision(FP, m::AbstractArray{<:AbstractFloat}) = FP(m)
_cast_to_precision(FP, m::AbstractFloat) = eltype(FP([1.0]))(m)

_cast_to_precision(FP, m::Ref{T}) where {T} = Ref{T}(_cast_to_precision(FP, m[]))

_cast_to_precision(FP, m::Tuple) = map(x -> _cast_to_precision(FP, x), m)
_cast_to_precision(FP, m::NamedTuple) = map(x -> _cast_to_precision(FP, x), m)

function _cast_to_precision(FP, m::AbstractGenerativeModel)
    T = typeof(m)
    fields = fieldnames(T)
    
    mapped_fields = map(fields) do f
        return _cast_to_precision(FP, getfield(m, f))
    end
    
    return T.name.wrapper(mapped_fields...)
end

_cast_to_precision(FP, m) = m

"""
$TYPEDEF

A protocol containing a user-specified generative model, and all of its 
parameters needed to train it and evaluate it. By default the training and
evaluation of the generative model is done in the CPU, however, this can be 
changed by setting `device` to a GPU device of preference.

# Structure Fields
$TYPEDFIELDS
"""
@kwdef struct GenerativeModelProtocol{M<:AbstractGenerativeModel}
    """Vector containing all the data, scaled and shifted to have zero mean and unit variance, that is to be used for training."""
    training_data::Union{Matrix, Nothing} = nothing
    """Mean of the raw (unshifted and unscaled) training data."""
    mean_training_data::Tuple
    """Variance of the raw (unshifted and unscaled) training data."""
    var_training_data::Tuple
    """Number of training epochs for the generative model."""
    epochs::Int = 100
    """Batch size for training the generative model."""
    batchsize::Int = 32
    """Flag indicating whether to shuffle the training data during training."""
    shuffle::Bool = true
    """Determines if the training data should be normalized to have zero mean and unit variance when passed to the generative model."""
    normalize_data::Bool = true
    """Optimizer used to train the generative model."""
    optimiser::Union{Any, Tuple} = Adam(0.01)
    """Hardware device (CPU or GPU) on which to perform training and inference."""
    device::Lux.MLDataDevices.AbstractDevice = cpu_device()
    """Generative model architecture to be used."""
    model::M
    """Floating point precision to be used"""
    precision::Function = f32
    """Stores the time-series data generated while training the generative model."""
    _log::Dict{String,Vector} = Dict{String,Vector}()

    function GenerativeModelProtocol(
    training_data::Union{Matrix, Nothing},
    mean_training_data::Tuple,
    var_training_data::Tuple,
    epochs::Int,
    batchsize::Int,
    shuffle::Bool,
    normalize_data::Bool,
    optimiser::Union{Any, Tuple},
    device::Lux.MLDataDevices.AbstractDevice,
    model::M,
    precision::Function,
    _log::Dict{String,Vector},
    ) where {M<:AbstractGenerativeModel}

        if normalize_data && !compatible_generative_protocol(training_data, var_training_data)
            @error "Incompatible GenerativeModelProtocol parameters!"
            throw(MethodError(GenerativeModelProtocol, (training_data, mean_training_data, var_training_data, epochs, batchsize, shuffle, normalize_data, optimiser, device, model, precision, _log)))
        end

        training_data = _cast_to_precision(precision, training_data)
        mean_training_data = _cast_to_precision(precision, mean_training_data)
        var_training_data = _cast_to_precision(precision, var_training_data)
        if optimiser isa Tuple
            optimiser = Tuple(_cast_to_precision(precision, opt) for opt in optimiser)
        else
            optimiser = _cast_to_precision(precision, optimiser)
        end
        model = _cast_to_precision(precision, model)

        M_c = typeof(model)

        return new{M_c}(training_data, mean_training_data, var_training_data, epochs, batchsize, shuffle, normalize_data, optimiser, device, model, precision, _log)
    end
end

GenerativeModelProtocol{M}(args...; kwargs...) where {M<:AbstractGenerativeModel} = GenerativeModelProtocol(args...; kwargs...)

"""
    GenerativeModelProtocol(model::M, training_data::Matrix{<:AbstractFloat}; 
        normalize_data::Bool = true, kwargs...) where {M<:AbstractGenerativeModel}

Convenience constructor to create a `GenerativeModelProtocol` with default 
training parameters, optimiser and compute device.

# Arguments
 - `model<:AbstractGenerativeModel`: The generative model to use.
 - `training_data::Matrix{<:AbstractFloat}`: The training data to use.

# Keyword Arguments
 - `normalize_data::Bool = true`: Whether to normalize the training data.
 - `kwargs...`: Additional keyword arguments for the generative model protocol.
"""
function GenerativeModelProtocol(model::M, training_data::Matrix{<:AbstractFloat}; normalize_data::Bool = true, kwargs...) where {M<:AbstractGenerativeModel}
    copy_training_data = copy(training_data)

    mean_training_data = mean(copy_training_data, dims = 2)
    var_training_data = var(copy_training_data, dims = 2)

    copy_training_data .-= mean_training_data
    copy_training_data ./= sqrt.(var_training_data)

    if normalize_data
        training_data = copy_training_data
    end

    return GenerativeModelProtocol(;
        training_data      = training_data,
        mean_training_data = Tuple(mean_training_data),
        var_training_data  = Tuple(var_training_data),
        model              = model,
        normalize_data     = normalize_data,
        kwargs...
    )
end

function _read_metadata end

function _pairwise_metric_distance end
function earth_mover_distance end
function sinkhorn_distance end
function energy_distance end

@compat public earth_mover_distance
@compat public sinkhorn_distance
@compat public energy_distance

macro _read_metadata(saved_protocol, main_group_name, metadata_group_name)
    return :(_read_metadata($(esc(saved_protocol)); 
        $(main_group_name = esc(main_group_name)), 
        $(metadata_group_name = esc(metadata_group_name))
    )) 
end

function GenerativeModelProtocol(saved_protocol::String;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name, 
    metadata_group_name::String = GenerativeModelProtocols.@default_metadata_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    model = nothing
    generative_model, mean_training_data, var_training_data, normalize_data = @_read_metadata(saved_protocol, main_group_name, metadata_group_name)

    function _gen_model(generative_model)
        if generative_model == variational_autoencoder
            return VariationalAutoencoder
        elseif generative_model == diffusion_model
            return DiffusionModel
        elseif generative_model == generative_adversarial_network
            return GenerativeAdversarialNetwork
        elseif generative_model == normalizing_flow
            return NormalizingFlow
        elseif generative_model == gaussian_mixture_model
            return GaussianMixtureModel
        end
    end

    model = _gen_model(generative_model)(saved_protocol;
        main_group_name             = main_group_name,
        generative_model_group_name = generative_model_group_name
    )

    return GenerativeModelProtocol(;
        model              = model,
        mean_training_data = mean_training_data,
        var_training_data  = var_training_data,
        normalize_data     = normalize_data
    )
end

macro _train!(protocol, model, print_log, kwargs...)
    :(_train!($(esc(protocol)), $(esc(model)); $(print_log = esc(print_log)), $(esc(kwargs...))))
end

"""
    train!(protocol::GenerativeModelProtocol; kwargs...)

Method to train the model inside `protocol` with its `training_data`. This 
method can be called again after it finishes to resume training.

# Arguments
 - `protocol`: The generative model protocol to train.

# Keyword Arguments (All Models)
 - `print_log::Bool`: Whether to print training logs (default is `true`).

# Keyword Arguments (VariationalAutoencoder)
 - `β::Union{Vector{<:AbstractFloat}, AbstractFloat}`: The β parameter for the VAE ELBO function (default is `1.0`).

# Keyword Arguments (GenerativeAdversarialNetwork)
 - `β::Union{Vector{<:AbstractFloat}, AbstractFloat}`: The β parameter for the VAE ELBO function (default is `1.0`).
 - `n_critic::Int`: The number of subiterations to train the critic in each training step (default is `5`).
 - `γ_vae::AbstractFloat`: The weight of the VAE ELBO in the VAE loss function (default is `1.0`).
 - `γ_wgan::AbstractFloat`: The weight of the Wasserstein distance in the VAE loss function (default is `1.0`).
 - `grad_penalty::Bool`: Whether to use gradient penalty in the WGAN loss function (default is `false`).
 - `λ::AbstractFloat`: The weight of the gradient penalty in the WGAN loss function (default is `10.0`).
 - `a::AbstractFloat`: The gradient norm target in the gradient penalty (default is `1.0`).
 - `weight_clipping::Bool`: Whether to use weight clipping in the WGAN loss function (default is `false`).
 - `clip_value::AbstractFloat`: The value to clip the weights and biases of the critic at each training step (default is `1.0`).
"""
function train!(protocol::GenerativeModelProtocol; print_log::Bool = true, kwargs...)
    empty!(protocol._log)

    t₀ = time()
    train_log = @_train!(protocol, protocol.model, print_log, kwargs...)
    t₁ = time()

    Δt = t₁ - t₀
    elapsed = canonicalize(Second(round(Int, Δt)))
    println("Training took $elapsed.")

    return (trace = train_log, elapsed = Δt)
end

function _shift_and_scale(protocol::GenerativeModelProtocol, x::Matrix)
    if protocol.normalize_data
        x = (x .- protocol.mean_training_data) ./ sqrt.(protocol.var_training_data)
    end

    return x
end

function _shift_and_scale(protocol::GenerativeModelProtocol, x::Vector)
    return _shift_and_scale(protocol, reshape(x, :, 1))[:]
end

function _unscale_and_unshift(protocol::GenerativeModelProtocol, x::Matrix)
    if protocol.normalize_data
        x = (x .* sqrt.(protocol.var_training_data)) .+ protocol.mean_training_data
    end

    return x
end

function _unscale_and_unshift(protocol::GenerativeModelProtocol, x::Vector)
    return _unscale_and_unshift(protocol, reshape(x, :, 1))[:]
end

function (protocol::GenerativeModelProtocol)(n_samples::Int; kwargs...)
    return _unscale_and_unshift(protocol, _eval(protocol.model, n_samples; kwargs...))
end

function (protocol::GenerativeModelProtocol)(; kwargs...)
    return _unscale_and_unshift(protocol, _eval(protocol.model, kwargs...))
end

function (protocol::GenerativeModelProtocol)(category_index::Int, n_samples::Int)
    return _unscale_and_unshift(protocol, _eval(protocol.model, category_index, n_samples))
end

"""
    categorize(protocol::GenerativeModelProtocol, x::Vector) -> Vector

Compute the posterior probability distribution over the mixture components for a 
given input vector `x`. Returns a vector where the k-th element represents the 
conditional probability that the input stems from the k-th categorical cluster 
of the model.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use for categorization.
 - `x::Vector`: The input vector for which to compute the posterior probability distribution.
"""
function categorize(protocol::GenerativeModelProtocol, x::Vector)::Vector
    return _categorize(protocol.model, _shift_and_scale(protocol, x))
end

"""
    categorize(protocol::GenerativeModelProtocol, x::Matrix) -> Matrix

Batch compute the posterior probability distributions over the mixture 
components for multiple input vectors. Each sample in the input matrix `x` is 
mapped to a normalized categorical probability vector where the k-th element 
represents the conditional probability that the sample stems from the k-th 
cluster.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use for categorization.
 - `x::Matrix`: The input matrix for which to compute the posterior probability distributions.
"""
function categorize(protocol::GenerativeModelProtocol, x::Matrix)::Matrix
    return _categorize(protocol.model, protocol.precision(_shift_and_scale(protocol, x)))
end

"""
    encode(protocol::GenerativeModelProtocol, x::Matrix; kwargs...) -> Matrix

Encodes the data space variable `x` into a latent space variable.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use for encoding.
 - `x::Matrix`: The input matrix to encode.

# Keyword Arguments (NormalizingFlow)
 - `ode_solver`: The ODE solver to use for the normalizing flow.
 - `t_final::AbstractFloat`: The final time for the ODE solver.
"""
function encode(protocol::GenerativeModelProtocol, x::Matrix; kwargs...)::Matrix
    return _encode(protocol.model, protocol.precision(_shift_and_scale(protocol, x)); kwargs...)
end

"""
    encode(protocol::GenerativeModelProtocol, x::Vector; kwargs...) -> Vector

Encodes the data space variable `x` into a latent space variable.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use for encoding.
 - `x::Vector`: The input vector to encode.

# Keyword Arguments (NormalizingFlow)
 - `ode_solver`: The ODE solver to use for the normalizing flow.
 - `t_final::AbstractFloat`: The final time for the ODE solver.
"""
function encode(protocol::GenerativeModelProtocol, x::Vector; kwargs...)::Vector
    return _encode(protocol.model, protocol.precision(_shift_and_scale(protocol, x)); kwargs...)
end

"""
    decode(protocol::GenerativeModelProtocol, z::Matrix; kwargs...) -> Matrix

Decodes the latent space representations `z` back into the data space.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use for decoding.
 - `z::Matrix`: The latent space representations to decode.

# Keyword Arguments (NormalizingFlow)
 - `ode_solver`: The ODE solver to use for the normalizing flow.
 - `t_final::AbstractFloat`: The final time for the ODE solver.
"""
function decode(protocol::GenerativeModelProtocol, z::Matrix; kwargs...)::Matrix
    return _unscale_and_unshift(protocol, protocol.precision(_decode(protocol.model, z; kwargs...)))
end

"""
    decode(protocol::GenerativeModelProtocol, z::Vector; kwargs...) -> Vector

Decodes the latent space representations `z` back into the data space.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use for decoding.
 - `z::Vector`: The latent space representations to decode.

# Keyword Arguments (NormalizingFlow)
 - `ode_solver`: The ODE solver to use for the normalizing flow.
 - `t_final::AbstractFloat`: The final time for the ODE solver.
"""
function decode(protocol::GenerativeModelProtocol, z::Vector; kwargs...)::Vector
    return _unscale_and_unshift(protocol, protocol.precision(_decode(protocol.model, z; kwargs...)))
end

"""
    input_size(protocol::GenerativeModelProtocol) -> Int

Returns the dimension size of an input vector.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use.
"""
function input_size(protocol::GenerativeModelProtocol)::Int
    return _input_size(protocol.model)
end

"""
    latent_size(protocol::GenerativeModelProtocol) -> Int

Returns the dimension size of an encoded latent vector.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use.
"""
function latent_size(protocol::GenerativeModelProtocol)::Int
    return _latent_size(protocol.model)
end

function _save_metadata end

macro _save_metadata(file, protocol, main_group_name, metadata_group_name, metadata)
    return :(_save_metadata($(esc(file)), $(esc(protocol)); 
        $(main_group_name = esc(main_group_name)),
        $(metadata_group_name = esc(metadata_group_name)),
        $(metadata = esc(metadata)),
    ))
end

function _save_model end

macro _save_model(file, model, main_group_name, generative_model_group_name)
    return :(_save_model($(esc(file)), $(esc(model)); 
        $(main_group_name = esc(main_group_name)),
        $(generative_model_group_name = esc(generative_model_group_name))
    ))
end

function _save(file::String, protocol::GenerativeModelProtocol; 
    main_group_name::String = @default_main_group_name,
    metadata_group_name::String = @default_metadata_group_name,
    generative_model_group_name::String = @default_generative_model_group_name,
    metadata::Union{Dict{String,Any}, NamedTuple, Nothing} = nothing)
    @_save_model(file, protocol.model, main_group_name, generative_model_group_name)
    @_save_metadata(file, protocol, main_group_name, metadata_group_name, metadata)
end

function Base.display(protocol::GenerativeModelProtocol)
    println("$(Base.typename(typeof(protocol)).wrapper):")
    println("training_data = $(summary(protocol.training_data))")
    println("epochs        = $(protocol.epochs)")
    println("batchsize     = $(protocol.batchsize)")
    println("shuffle       = $(protocol.shuffle)")
    println("optimiser     = $(protocol.optimiser)")
    println("device        = $(nameof(protocol.device))")
    println("model         = $(Base.typename(typeof(protocol.model)).wrapper)")
end

# Auxiliary scripts
include("auxiliary/activation_functions.jl")
include("auxiliary/input_output_sizes.jl")
include("auxiliary/printing.jl")
include("auxiliary/load_data.jl")
include("auxiliary/optimisation.jl")
include("auxiliary/named_tuples.jl")
include("auxiliary/tabular_denoiser.jl")

# Generative models
include("diffusion_model.jl")
include("normalizing_flow.jl")
include("gaussian_mixture_model.jl")
include("variational_autoencoder.jl")
include("generative_adversarial_network.jl")

end

