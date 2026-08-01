module GenerativeModelProtocols

using Flux
using Printf
using Zygote
using MLUtils
using StatsBase
using LinearAlgebra
using DocStringExtensions

using Compat: @compat

export GenerativeModelProtocol, train!, categorize, encode, decode

abstract type AbstractGenerativeModel end
abstract type AbstractCategoricalGenerativeModel <: AbstractGenerativeModel end

@kwdef struct TrainingLog
    loss::Vector{Float64} = Float64[]
    loss_grad_norm::Vector{Float64} = Float64[]
end

"""
$TYPEDEF

A protocol containing a user-specified generative model, and all of its 
parameters needed to train it and evaluate it. By default the training and
evaluation of the generative model is done in the CPU, however, this can be 
changed by setting `device` to a GPU device of preference.

$TYPEDFIELDS
"""
@kwdef struct GenerativeModelProtocol{M<:AbstractGenerativeModel}
    """Vector containing all the data that is to be used for training."""
    training_data::Union{Matrix{Float64}, Nothing}
    """Number of training epochs for the generative model."""
    epochs::Int = 100
    """Batch size for training the generative model."""
    batchsize::Int = 32
    """Flag indicating whether to shuffle the training data during training."""
    shuffle::Bool = true
    """Optimizer used to train the generative model."""
    optimiser::Union{Optimisers.AbstractRule, Flux.Optimise.AbstractOptimiser} = Adam(0.01f0)
    """Hardware device (CPU or GPU) on which to perform training and inference."""
    device::Flux.MLDataDevices.AbstractDevice = Flux.cpu_device()
    """Generative model architecture to be used."""
    model::M
    """Stores the time-series data generated while training the generative model."""
    _log::TrainingLog = TrainingLog()
end

"""
    GenerativeModelProtocol(model::M, training_data::Union{Matrix{Float64}, Nothing}; kwargs...) where {M<:AbstractGenerativeModel}

Convenience constructor to create a `GenerativeModelProtocol` with default 
training parameters, optimiser and compute device.
"""
function GenerativeModelProtocol(model::M, training_data::Union{Matrix{Float64}, Nothing} = nothing; kwargs...) where {M<:AbstractGenerativeModel}
    return GenerativeModelProtocol{M}(;
        training_data = training_data,
        model = model,
        kwargs...
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

    @_train!(protocol, protocol.model, print_log, kwargs...)
end

function (protocol::GenerativeModelProtocol)(n_samples::Int)
    return  protocol.model(n_samples)
end

function (protocol::GenerativeModelProtocol)()
    return protocol.model()
end

function (protocol::GenerativeModelProtocol)(category_index::Int, n_samples::Int)
    return protocol.model(category_index, n_samples)
end

"""
    categorize(protocol::GenerativeModelProtocol, x::Vector) -> Vector

Compute the posterior probability distribution over the mixture components for a 
given input vector `x`. Returns a vector where the k-th element represents the 
conditional probability that the input stems from the k-th categorical cluster 
of the model.
"""
function categorize(protocol::GenerativeModelProtocol, x::Vector)::Vector
    return categorize(protocol.model, x)
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
    return categorize(protocol.model, x)
end

"""
    encode(protocol::GenerativeModelProtocol, x::Matrix) -> Matrix

Encodes the data space variable `x` into a latent space variable.
"""
function encode(protocol::GenerativeModelProtocol, x::Matrix)::Matrix
    return encode(protocol.model, x)
end

"""
    encode(protocol::GenerativeModelProtocol, x::Vector) -> Vector

Encodes the data space variable `x` into a latent space variable.
"""
function encode(protocol::GenerativeModelProtocol, x::Vector)::Vector
    return encode(protocol.model, x)
end

"""
    decode(protocol::GenerativeModelProtocol, z::Matrix) -> Matrix

Decodes the latent space representations `z` back into the data space.
"""
function decode(protocol::GenerativeModelProtocol, z::Matrix)::Matrix
    return decode(protocol.model, z)
end

"""
    decode(protocol::GenerativeModelProtocol, z::Vector) -> Vector

Decodes the latent space representations `z` back into the data space.
"""
function decode(protocol::GenerativeModelProtocol, z::Vector)::Vector
    return decode(protocol.model, z)
end

function load_data(data::Matrix, batchsize::Int, shuffle::Bool = true, parallel::Bool = true)
    data = shuffle ? shuffleobs(data) : data

    return Flux.DataLoader(
        data, 
        batchsize = batchsize, 
        shuffle = false,
        parallel = true
    )
end

function load_data(data::Tuple{Vararg{Matrix}}, batchsize::Int, shuffle::Bool = true, parallel::Bool = true)
    data = shuffle ? shuffleobs(data) : data

    return Flux.DataLoader(
        data, 
        batchsize = batchsize, 
        shuffle = false,
        parallel = true
    )
end

@enum GenerativeModel::UInt8 begin
    variational_autoencoder         = 0
    diffusion_model                 = 1
    generative_adversarial_network  = 2
    normalizing_flow                = 3
    gaussian_mixture_model          = 4
end

macro _generative_model(model)
    return :(_generative_model($(esc(model))))
end

function _generative_model(protocol::GenerativeModelProtocol)::GenerativeModel
    return @_generative_model(protocol.model)
end

@enum ActivationFunction::UInt8 begin
    _relu       = 0
    _gelu       = 1
    _tanh       = 2
    _sigmoid    = 3
    _softmax    = 4
    _leakyrelu  = 5
    _elu        = 6
    _swish      = 7
    _identity   = 8
end

function activation_function_map()
    return Dict(
        relu        => UInt8(_relu),
        gelu        => UInt8(_gelu),
        tanh        => UInt8(_tanh),
        sigmoid     => UInt8(_sigmoid),
        softmax     => UInt8(_softmax),
        leakyrelu   => UInt8(_leakyrelu),
        elu         => UInt8(_elu),
        swish       => UInt8(_swish),
        identity    => UInt8(_identity)
    )
end

function activation_function_inverse_map()
    foo_map = activation_function_map()
    return Dict(foo_map[key] => key for key in keys(foo_map))
end

function compatible_neural_networks(networks::Tuple{Vararg{Chain}})
    for network in networks
        if !compatible_neural_network(network)
            return false
        end
    end

    return true
end

function compatible_neural_network(network::Chain)
    for layer in network
        if layer.σ ∉ keys(activation_function_map())
            return false
        end
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

macro _default_print_padding()
    return "   "
end

function _base_print_layers(layers::Tuple{Vararg{Dense}}, print_padding::String = @_default_print_padding)
    for layer in layers
        print(print_padding * print_padding)
        println(layer)
    end
end

function _print_chains(chains::Tuple{Chain}, print_padding::String = @_default_print_padding)
    for i in eachindex(chains)
        println(print_padding * "Chain(")
        _base_print_layers(chains[i].layers)
        println(print_padding * ")")
    end
end

function _print_layers(layers::Tuple{Vararg{Dense}}, print_padding::String = @_default_print_padding)
    println(print_padding * "Tuple()")
    _base_print_layers(layers)
    println(print_padding * ")")
end

function _print_chains(chain::Chain, print_padding)
    _print_chains((chain,), print_padding)
end

function input_size(layer::Dense)
    return size(layer.weight, 2)
end

function input_size(chain::Chain)
    return input_size(chain[1])
end

function output_size(layer::Dense)
    return size(layer.weight, 1)
end

function output_size(chain::Chain)
    return output_size(chain[end])
end

function Base.display(protocol::GenerativeModelProtocol)
    println("$(summary(protocol)):")
    println("training_data = $(summary(protocol.training_data))")
    println("epochs        = $(protocol.epochs)")
    println("batchsize     = $(protocol.batchsize)")
    println("shuffle       = $(protocol.shuffle)")
    println("optimiser     = $(protocol.optimiser)")
    println("device        = $(nameof(protocol.device))")
    println("model         = $(summary(protocol.model))")
end

# Auxiliary scripts
include("tabular_denoiser.jl")

# Generative models
include("diffusion_model.jl")
include("gaussian_mixture_model.jl")
include("variational_autoencoder.jl")

end

