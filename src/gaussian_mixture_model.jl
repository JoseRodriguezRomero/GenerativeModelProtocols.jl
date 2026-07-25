macro _gmm_default_activation_function()
    return relu
end

function _gmm_default_default_predictor_network(input_size::Int, k::Int, hidden_layer_size::Int = 32, activation_function::Function = @_gmm_default_activation_function)
    return Chain(
        Dense(input_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => k)
    ) |> f64
end

"""
$TYPEDEF

A structure containing the general parameters needed to evaluate and train 
a Gaussian Mixture Model (GMM). Once trained, it can be used a generative 
model.

$TYPEDFIELDS
"""
@kwdef struct GaussianMixtureModel <: AbstractGenerativeModel
    """Number Gaussian clusters in the mixture model."""
    k::Int
    """Logarithm of the square of the variance of the Gaussian clusters of the model."""
    log_σ²::Matrix{Float64}
    """Mean values of the Gaussian clusters of the model."""
    μ::Matrix{Float64}
    """Neural network parametrizing the likelihood of some input stemming from a given cluster."""
    predictor_network::Chain
end

"""
    GenerativeModelProtocols.GaussianMixtureModel(input_size::Int, k::Int)

Convenience constructor that generates a 
`GenerativeModelProtocols.GaussianMixtureModel` using default predictor network 
architecture.
"""
function GaussianMixtureModel(input_size::Int, k::Int)
    return GaussianMixtureModel(;
        k                   = k,
        log_σ²              = zeros(Float64,k,input_size),
        μ                   = zeros(Float64,k,input_size),
        predictor_network   = _gmm_default_default_predictor_network(input_size,k)
    )
end

function Base.display(model::GaussianMixtureModel)
    print_padding = @_default_print_padding
    println("GenerativeModelProtocols.GaussianMixtureModel:")
    println("k      = $(model.β)")
    println("μ      = $(size(model.μ,1))×$(size(model.μ,2)) Matrix{Float64}")
    println("log_σ² = $(size(model.log_σ²,1))×$(size(model.log_σ²,2)) Matrix{Float64}")
    println("")

    function print_chains(chains)
        for i in eachindex(chains)
            println(print_padding * "Chain(")
            for layer in chains[i]
                print(print_padding * print_padding)
                println(layer)
            end
            println(print_padding * ")")
        end
    end

    println("predictor_network: ")
    _print_chains(model.predictor_network, print_padding)
end

function load_gaussian_mixture_parameters(saved_model::Any;
    main_group_name::String = @default_main_group_name,
    generative_model_group_name::String = @default_generative_model_group_name)
    throw(ArgumentError("Types $(typeof(saved_model)) does not implement the required `load_gaussian_mixture_parameters` interface."))
end

function _generative_model(::GaussianMixtureModel)::GenerativeModel
    return gaussian_mixture_model
end

