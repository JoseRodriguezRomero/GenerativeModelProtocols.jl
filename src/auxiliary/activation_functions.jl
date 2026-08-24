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

