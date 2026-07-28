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

macro _train!(protocol, model, print_log)
    :(_train!($(esc(protocol)), $(esc(model)); $(print_log = esc(print_log))))
end

function _train!(protocol::GenerativeModelProtocol, model::AbstractGenerativeModel; print_log::Bool = true)
    @_train!(protocol, model, print_log)
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
    train!(protocol::GenerativeModelProtocol; print_log::Bool = true)

Method to train the model inside `protocol` with its `training_data`. This 
method can be called again after it finishes to resume training.
"""
function train!(protocol::GenerativeModelProtocol; print_log::Bool = true)
    clean_logged_data!(protocol._log, protocol.epochs)

    if isnothing(protocol.training_data)
        throw(ArgumentError("No training data was loaded!"))
    end

    _train!(protocol, protocol.model; print_log = print_log)
end

macro _eval(protocol, model, n_samples)
    return :(_eval($(esc(protocol)), $(esc(model)), $(esc(n_samples))))
end

macro _categorical_eval(protocol, model, category_index, n_samples)
    return :(_categorical_eval($(esc(protocol)), $(esc(model)), $(esc(category_index)), $(esc(n_samples))))
end

macro _categorize(protocol, model, x)
    return :(_categorize($(esc(protocol)), $(esc(model)), $(esc(x))))
end

function (protocol::GenerativeModelProtocol)(n_samples::Int)
    return @_eval(protocol, protocol.model, n_samples)
end

function (protocol::GenerativeModelProtocol)(category_index::Int, n_samples::Int)
    return @_categorical_eval(protocol, protocol.model, category_index, n_samples)
end

"""
    categorize(protocol::GenerativeModelProtocol, x::Vector{Float64}) -> Vector{Float64}

Compute the posterior probability distribution over the mixture components for a 
given input vector `x`. Returns a vector where the k-th element represents the 
conditional probability that the input stems from the k-th categorical cluster 
of the model.
"""
function categorize(protocol::GenerativeModelProtocol, x::Vector{Float64})::Vector{Float64}
    return @_categorize(protocol, protocol.model, x)
end

"""
    categorize(protocol::GenerativeModelProtocol, x::Matrix{Float64}) -> Matrix{Float64}

Batch compute the posterior probability distributions over the mixture 
components for multiple input vectors. Each sample in the input matrix `x` is 
mapped to a normalized categorical probability vector where the k-th element 
represents the conditional probability that the sample stems from the k-th 
cluster.
"""
function categorize(protocol::GenerativeModelProtocol, x::Matrix{Float64})::Matrix{Float64}
    return @_categorize(protocol, protocol.model, x)
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
        relu        => _relu,
        gelu        => _gelu,
        tanh        => _tanh,
        sigmoid     => _sigmoid,
        softmax     => _softmax,
        leakyrelu   => _leakyrelu,
        elu         => _elu,
        swish       => _swish,
        identity    => _identity
    )
end

function compatible_neural_networks(networks::Tuple{Vararg{Chain}})
    for network in networks
        if !compatible_neural_network(network)
            return false
        end
    end

    return true
end

function compatible_neural_networks(networks::Vector{Chain})
    return compatible_neural_networks(Tuple(networks))
end

function compatible_neural_network(network::Chain)
    for layer in network
        if layer.σ ∉ keys(activation_function_map())
            return false
        end
    end

    return true
end

function activation_function_inverse_map()
    foo_map = activation_function_map()
    return Dict(foo_map[key] => key for key in keys(foo_map))
end

function enum_activation_function_map()
    return Dict(f => UInt8(f) for f in instances(ActivationFunction))
end

function enum_activation_function_inverse_map()
    return Dict(UInt8(f) => f for f in instances(ActivationFunction))
end

struct LayerParameters
    node_bias::Vector{Float64}
    node_weights::Matrix{Float64}
    activation_function::ActivationFunction
end

struct ChainParameters
    layers::Vector{LayerParameters}
end

function layer_parameters(layer::Dense)::LayerParameters
    foo_map = activation_function_map()

    if hasproperty(layer,:σ)
        return LayerParameters(
            layer.bias,
            layer.weight,
            foo_map[layer.σ]
        )
    end

    return LayerParameters(
        layer.bias,
        layer.weight,
        foo_map[identity]
    )
end

function layer_parameters(layer::LayerParameters)::Dense
    foo_inv_map = activation_function_inverse_map()
    return Dense(layer.node_weights, layer.node_bias, foo_inv_map[layer.activation_function])
end

function chain_parameters(chain::Chain)::ChainParameters
    layers = Vector{LayerParameters}(undef, length(chain))
    for i in eachindex(chain)
        layers[i] = layer_parameters(chain[i])
    end

    return ChainParameters(layers)
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

function _print_chains(chains::Tuple{Chain}, print_padding = @_default_print_padding)
    for i in eachindex(chains)
        println(print_padding * "Chain(")
        for layer in chains[i]
            print(print_padding * print_padding)
            println(layer)
        end
        println(print_padding * ")")
    end
end

function _print_chains(chains::Vector{Chain}, print_padding = @_default_print_padding)
    _print_chains(Tuple(chains), print_padding)
end

function _print_chains(chain::Chain, print_padding)
    _print_chains((chain,), print_padding)
end

function Base.display(protocol::GenerativeModelProtocol)
    device_flag_map = Dict(
        variational_autoencoder         => "VariationalAutoencoder",
        diffusion_model                 => "DiffusionModel",
        generative_adversarial_network  => "GenerativeAdversationNetwork",
        normalizing_flow                => "NormalizingFlow",
        gaussian_mixture_model          => "GaussianMixtureModel"
    )

    println("GenerativeModelProtocol:")
    if isnothing(protocol.training_data)
        println("training_data = nothing")
    else
        println("training_data = $(size(protocol.training_data,1))×$(size(protocol.training_data,2)) Matrix{Float64}")
    end
    println("epochs        = $(protocol.epochs)")
    println("batchsize     = $(protocol.batchsize)")
    println("shuffle       = $(protocol.shuffle)")
    println("optimiser     = $(protocol.optimiser)")
    println("device        = $(nameof(protocol.device))")
    println("model         = $(device_flag_map[_generative_model(protocol.model)])")
end

include("diffusion_model.jl")
include("gaussian_mixture_model.jl")
include("variational_autoencoder.jl")

end

