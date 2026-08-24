module GenerativeModelProtocols

using Flux
using Printf
using Enzyme
using MLUtils
using StatsBase
using LinearAlgebra
using DocStringExtensions

using Compat: @compat

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

    ϵ = 1.0E-6
    if !isnothing(training_data)
        if maximum(abs.(mean(training_data, dims = 2))) > ϵ
            println("A")
            println(maximum(abs.(mean(training_data, dims = 2))))
            print("-----------------------------------")
            return false
        end

        if maximum(abs.(var(training_data, dims = 2) .- 1.0)) > ϵ
            println("B")
            println(maximum(abs.(var(training_data, dims = 2) .- 1.0)))
            print("-----------------------------------")
            return false
        end
    end

    if minimum(var_training_data) < ϵ
        println("C")
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
    """Optimizer used to train the generative model."""
    optimiser::Union{O, Tuple{Vararg{O}}} = Adam(0.01)
    """Hardware device (CPU or GPU) on which to perform training and inference."""
    device::Flux.MLDataDevices.AbstractDevice = Flux.cpu_device()
    """Generative model architecture to be used."""
    model::M
    """Stores the time-series data generated while training the generative model."""
    _log::TrainingLog = TrainingLog()
    """Floating point precision to be used"""
    precision::Function = f32

    function GenerativeModelProtocol(
        training_data::Union{Matrix, Nothing},
        mean_training_data::Tuple,
        var_training_data::Tuple,
        epochs::Int,
        batchsize::Int,
        shuffle::Bool,
        optimiser::Union{O, Tuple{Vararg{O}}},
        device::Flux.MLDataDevices.AbstractDevice,
        model::M,
        _log::TrainingLog,
        precision::Function
        ) where {M<:AbstractGenerativeModel, O}

        if !compatible_generative_protocol(training_data, var_training_data)
            @error "Incompatible GenerativeModelProtocol parameters!"
            throw(MethodError(GenerativeModelProtocol, (training_data, mean_training_data, var_training_data, epochs, batchsize, shuffle, optimiser, device, model, _log)))
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

        return new{M_c,O_c}(training_data, mean_training_data, var_training_data, epochs, batchsize, shuffle, optimiser, device, model, _log, precision)
    end
end

GenerativeModelProtocol{M}(args...; kwargs...) where {M<:AbstractGenerativeModel} = GenerativeModelProtocol(args...; kwargs...)

"""
    GenerativeModelProtocol(model::M, training_data::Matrix{<:AbstractFloat}; precision::Function = f32, kwargs...) where {M<:AbstractGenerativeModel}

Convenience constructor to create a `GenerativeModelProtocol` with default 
training parameters, optimiser and compute device.
"""
function GenerativeModelProtocol(model::M, training_data::Matrix{<:AbstractFloat}; precision::Function = f32, kwargs...) where {M<:AbstractGenerativeModel}
    copy_training_data = copy(training_data)

    mean_training_data = mean(copy_training_data, dims = 2)
    var_training_data = var(copy_training_data, dims = 2)

    copy_training_data .-= mean_training_data
    copy_training_data ./= sqrt.(var_training_data)

    return GenerativeModelProtocol{M}(;
        training_data      = copy_training_data,
        mean_training_data = Tuple(mean_training_data),
        var_training_data  = Tuple(var_training_data),
        model              = model,
        precision          = precision,
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
    generative_model, mean_training_data, var_training_data = @_read_metadata(saved_protocol, main_group_name, metadata_group_name)

    function _gen_model(generative_model)
        if generative_model == variational_autoencoder
            return VariationalAutoencoder
        elseif generative_model == diffusion_model
            return DiffusionModel
        elseif generative_model == generative_adversarial_network
            return GenerativeAdversarialNetwork
        elseif generative_model == normalizing_flow
            return nothing # Temporary dummy line
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
        var_training_data  = var_training_data
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

function _shift_and_scale(protocol::GenerativeModelProtocol, x::Matrix)
    return (x .- protocol.mean_training_data) ./ sqrt.(protocol.var_training_data)
end

function _shift_and_scale(protocol::GenerativeModelProtocol, x::Vector)
    return _shift_and_scale(protocol, reshape(x, :, 1))[:]
end

function _unscale_and_unshift(protocol::GenerativeModelProtocol, x::Matrix)
    return (x .* sqrt.(protocol.var_training_data)) .+ protocol.mean_training_data
end

function _unscale_and_unshift(protocol::GenerativeModelProtocol, x::Vector)
    return _unscale_and_unshift(protocol, reshape(x, :, 1))[:]
end

function (protocol::GenerativeModelProtocol)(n_samples::Int)
    return _unscale_and_unshift(protocol, _eval(protocol.model, n_samples))
end

function (protocol::GenerativeModelProtocol)()
    return _unscale_and_unshift(protocol, _eval(protocol.model))
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
    encode(protocol::GenerativeModelProtocol, x::Matrix) -> Matrix

Encodes the data space variable `x` into a latent space variable.
"""
function encode(protocol::GenerativeModelProtocol, x::Matrix)::Matrix
    return _encode(protocol.model, protocol.precision(_shift_and_scale(protocol, x)))
end

"""
    encode(protocol::GenerativeModelProtocol, x::Vector) -> Vector

Encodes the data space variable `x` into a latent space variable.
"""
function encode(protocol::GenerativeModelProtocol, x::Vector)::Vector
    return _encode(protocol.model, protocol.precision(_shift_and_scale(protocol, x)))
end

"""
    decode(protocol::GenerativeModelProtocol, z::Matrix) -> Matrix

Decodes the latent space representations `z` back into the data space.
"""
function decode(protocol::GenerativeModelProtocol, z::Matrix)::Matrix
    return _unscale_and_unshift(protocol, protocol.precision(_decode(protocol.model, z)))
end

"""
    decode(protocol::GenerativeModelProtocol, z::Vector) -> Vector

Decodes the latent space representations `z` back into the data space.
"""
function decode(protocol::GenerativeModelProtocol, z::Vector)::Vector
    return _unscale_and_unshift(protocol, protocol.precision(_decode(protocol.model, z)))
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

    return Flux.DataLoader(
        data, 
        batchsize = batchsize, 
        shuffle = false,
        parallel = true
    )
end

function load_data(data::Tuple{Vararg{AbstractMatrix}}, batchsize::Int, shuffle::Bool = true, parallel::Bool = true)
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

function generative_model_map()
    return Dict(
        UInt8(variational_autoencoder)        => variational_autoencoder,
        UInt8(diffusion_model)                => diffusion_model,
        UInt8(generative_adversarial_network) => generative_adversarial_network,
        UInt8(normalizing_flow)               => normalizing_flow,
        UInt8(gaussian_mixture_model)         => gaussian_mixture_model
    )
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

function compatible_neural_networks(networks::Tuple{Vararg{C}}) where C
    for network in networks
        if !compatible_neural_network(network)
            return false
        end
    end

    return true
end

function compatible_neural_network(network::C) where C
    for layer in network
        if layer.σ ∉ keys(activation_function_map())
            return false
        end
    end

    return true
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

function _base_print_layers(layers::Tuple{Vararg{<:Dense}}, print_padding::String = @_default_print_padding)
    for layer in layers
        print(print_padding * print_padding)
        println(layer)
    end
end

function _print_chains(chains::Tuple{Vararg{<:Chain}}, print_padding::String = @_default_print_padding)
    for i in eachindex(chains)
        println(print_padding * "Chain(")
        _base_print_layers(chains[i].layers)
        println(print_padding * ")")
    end
end

function _print_layers(layers::Tuple{Vararg{<:Dense}}, print_padding::String = @_default_print_padding)
    println(print_padding * "Tuple()")
    _base_print_layers(Tuple(layers))
    println(print_padding * ")")
end

function _print_chains(chain::C, print_padding) where {C<:Chain}
    _print_chains((chain,), print_padding)
end

function _input_size(layer::L) where {L<:Dense}
    return size(layer.weight, 2)
end

function _input_size(chain::C) where {C<:Chain}
    return _input_size(chain[1])
end

function _output_size(layer::L) where {L<:Dense}
    return size(layer.weight, 1)
end

function _output_size(chain::C) where {C<:Chain}
    return _output_size(chain[end])
end

function load_model!(dst::Tuple{Vararg{<:Chain}}, src::Tuple{Vararg{<:Chain}})
    for i in eachindex(dst)
        Flux.loadmodel!(dst[i], src[i])
    end
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
include("generative_adversarial_network.jl")

end

