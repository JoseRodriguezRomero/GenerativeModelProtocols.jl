module GenerativeModelProtocols

using Lux
using Printf
using Enzyme
using Random
using MLUtils
using StatsBase
using Optimisers
using LinearAlgebra
using DocStringExtensions

using Compat: @compat
using EnzymeCore

export GenerativeModelProtocol, train!, categorize, encode, decode, input_size, latent_size

abstract type AbstractGenerativeModel end
abstract type AbstractCategoricalGenerativeModel <: AbstractGenerativeModel end

@kwdef struct TrainingLog
    loss::Vector{Float64} = Float64[]
    loss_grad_norm::Vector{Float64} = Float64[]
end

function compatible_generative_protocol(
    training_data::Union{Matrix{F}, Nothing},
    var_training_data::Tuple{Vararg{F}}
    ) where {F<:AbstractFloat}

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

function _cast_to_precision(FP, m)
    FT = eltype(FP([1.0]))
    
    if m isa AbstractArray{<:AbstractFloat}
        return FP(m)
    elseif m isa Tuple
        return tuple((_cast_to_precision(FP, mi) for mi in m)...)
    elseif m isa Array{Any,0}
        return Array{Any,0}(fill(FP(m[1])))
    elseif m isa NamedTuple
        fields = keys(m)
        mapped_values = map(fields) do f
            return _cast_to_precision(FP, getfield(m, f))
        end
        
        return NamedTuple{fields}(Tuple(mapped_values))
    elseif !isprimitivetype(typeof(m)) && fieldcount(typeof(m)) > 0
        fields = fieldnames(typeof(m))
        mapped_values = map(fields) do f
            val = getfield(m, f)
            return _cast_to_precision(FP, val)
        end

        return typeof(m).name.wrapper(mapped_values...)
    elseif m isa AbstractFloat
        return FT(m)
    else
        return m
    end
end

"""
$TYPEDEF

A protocol containing a user-specified generative model, and all of its 
parameters needed to train it and evaluate it. By default the training and
evaluation of the generative model is done in the CPU, however, this can be 
changed by setting `device` to a GPU device of preference.

$TYPEDFIELDS
"""
@kwdef struct GenerativeModelProtocol{M<:AbstractGenerativeModel, O}
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
    optimiser::Union{O, Tuple{Vararg{O}}} = Adam(0.01)
    """Hardware device (CPU or GPU) on which to perform training and inference."""
    device::Lux.MLDataDevices.AbstractDevice = cpu_device()
    """Generative model architecture to be used."""
    model::M
    """Floating point precision to be used"""
    precision::Function = f32
    """Stores the time-series data generated while training the generative model."""
    _log::TrainingLog = TrainingLog()

    function GenerativeModelProtocol(
    training_data::Union{Matrix, Nothing},
    mean_training_data::Tuple,
    var_training_data::Tuple,
    epochs::Int,
    batchsize::Int,
    shuffle::Bool,
    normalize_data::Bool,
    optimiser::Union{O, Tuple{Vararg{O}}},
    device::Lux.MLDataDevices.AbstractDevice,
    model::M,
    precision::Function,
    _log::TrainingLog,
    ) where {M<:AbstractGenerativeModel, O}

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
        O_c = typeof(optimiser)

        return new{M_c,O_c}(training_data, mean_training_data, var_training_data, epochs, batchsize, shuffle, normalize_data, optimiser, device, model, precision, _log)
    end
end

GenerativeModelProtocol{M,O}(args...; kwargs...) where {M<:AbstractGenerativeModel, O} = GenerativeModelProtocol(args...; kwargs...)

"""
    GenerativeModelProtocol(model::M, training_data::Matrix{<:AbstractFloat}; normalize_data::Bool = true, kwargs...) where {M<:AbstractGenerativeModel}

Convenience constructor to create a `GenerativeModelProtocol` with default 
training parameters, optimiser and compute device.
"""
function GenerativeModelProtocol(model::M, training_data::Matrix{<:AbstractFloat}; normalize_data::Bool = true, kwargs...) where {M<:AbstractGenerativeModel, O}
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

function clean_logged_data!(log::Vector{Float64}, epochs::Int)
    resize!(log, epochs)
    log[:] .= Float64(0.0)
end

function clean_logged_data!(log::TrainingLog, epochs::Int)
    clean_logged_data!(log.loss, epochs)
    clean_logged_data!(log.loss_grad_norm, epochs)
end

"""
    train!(protocol::GenerativeModelProtocol; kwargs...)

Method to train the model inside `protocol` with its `training_data`. This 
method can be called again after it finishes to resume training.
"""
function train!(protocol::GenerativeModelProtocol; print_log::Bool = true, kwargs...)
    clean_logged_data!(protocol._log, protocol.epochs)

    if isnothing(protocol.training_data) || isempty(protocol.training_data)
        throw(ArgumentError("No training data was loaded!"))
    end

    local train_log
    if protocol.device isa CPUDevice
        # Loop unrolling is disabled until #3568 of Enzyme.jl is fixed.
        Enzyme.Compiler.LLVM.clopts("-unroll-runtime=false")
        train_log = @_train!(protocol, protocol.model, print_log, kwargs...)
        Enzyme.Compiler.LLVM.clopts("-unroll-runtime=true")
    else
        train_log = @_train!(protocol, protocol.model, print_log, kwargs...)
    end

    return train_log
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
"""
function categorize(protocol::GenerativeModelProtocol, x::Matrix)::Matrix
    return _categorize(protocol.model, protocol.precision(_shift_and_scale(protocol, x)))
end

"""
    encode(protocol::GenerativeModelProtocol, x::Matrix; kwargs...) -> Matrix

Encodes the data space variable `x` into a latent space variable.
"""
function encode(protocol::GenerativeModelProtocol, x::Matrix; kwargs...)::Matrix
    return _encode(protocol.model, protocol.precision(_shift_and_scale(protocol, x)); kwargs...)
end

"""
    encode(protocol::GenerativeModelProtocol, x::Vector; kwargs...) -> Vector

Encodes the data space variable `x` into a latent space variable.
"""
function encode(protocol::GenerativeModelProtocol, x::Vector; kwargs...)::Vector
    return _encode(protocol.model, protocol.precision(_shift_and_scale(protocol, x)); kwargs...)
end

"""
    decode(protocol::GenerativeModelProtocol, z::Matrix; kwargs...) -> Matrix

Decodes the latent space representations `z` back into the data space.
"""
function decode(protocol::GenerativeModelProtocol, z::Matrix; kwargs...)::Matrix
    return _unscale_and_unshift(protocol, protocol.precision(_decode(protocol.model, z; kwargs...)))
end

"""
    decode(protocol::GenerativeModelProtocol, z::Vector; kwargs...) -> Vector

Decodes the latent space representations `z` back into the data space.
"""
function decode(protocol::GenerativeModelProtocol, z::Vector; kwargs...)::Vector
    return _unscale_and_unshift(protocol, protocol.precision(_decode(protocol.model, z; kwargs...)))
end

"""
    input_size(protocol::GenerativeModelProtocol) -> Int

Returns the dimension size of an input vector.
"""
function input_size(protocol::GenerativeModelProtocol)::Int
    return _input_size(protocol.model)
end

"""
    latent_size(protocol::GenerativeModelProtocol) -> Int

Returns the dimension size of an encoded latent vector.
"""
function latent_size(protocol::GenerativeModelProtocol)::Int
    return _latent_size(protocol.model)
end

function load_data(data::AbstractMatrix, batchsize::Int, shuffle::Bool = true, parallel::Bool = true)
    data = shuffle ? shuffleobs(data) : data

    return MLUtils.DataLoader(
        data, 
        batchsize = batchsize, 
        shuffle = false,
        parallel = true
    )
end

function load_data(data::Tuple{Vararg{AbstractMatrix}}, batchsize::Int, shuffle::Bool = true, parallel::Bool = true)
    data = shuffle ? shuffleobs(data) : data

    return DataLoader(
        data, 
        batchsize = batchsize, 
        shuffle = false,
        parallel = true
    )
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

