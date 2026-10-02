module GenerativeModelProtocolsTulipExt

import GenerativeModelProtocols

using Distances
using OptimalTransport
using Tulip

"""
    GenerativeModelProtocols.earth_mover_distance(
        synthetic_data::AbstractMatrix, target_data::AbstractMatrix; 
        metric = Distances.Euclidean(),max_iter::Int = 5000)

Calculates the Earth Mover's Distance between synthetic data and target data.

# Arguments
 - `synthetic_data::AbstractMatrix`: A matrix of synthetic data points
 - `target_data::AbstractMatrix`: A matrix of target data points

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
 - `max_iter::Int`: Maximum number of iterations for the LP optimizer (default is 5000)
"""
function GenerativeModelProtocols.earth_mover_distance(
    synthetic_data::AbstractMatrix, target_data::AbstractMatrix; 
    metric = Distances.Euclidean(),
    max_iter::Int = 5000)

    num_samples_target = size(target_data, 2)
    num_samples_synthetic = size(synthetic_data, 2)

    μ = fill(1.0 / num_samples_target, num_samples_target)
    ν = fill(1.0 / num_samples_synthetic, num_samples_synthetic)
    C = GenerativeModelProtocols._pairwise_metric_distance(synthetic_data, target_data, metric)
    
    lp_optimizer = Tulip.Optimizer()
    Tulip.MOI.set(lp_optimizer, Tulip.MOI.RawOptimizerAttribute("IPM_IterationsLimit"), max_iter)

    return emd2(μ, ν, Float64.(C), lp_optimizer)
end

"""
    GenerativeModelProtocols.earth_mover_distance(
        protocol::GenerativeModelProtocol, target_data::AbstractMatrix; 
        metric = Distances.Euclidean(), max_iter::Int = 5000)

Calculates the Earth Mover's Distance between synthetic data produced by the 
protocol and target data.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use
 - `target_data::AbstractMatrix`: A matrix of target data points

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
 - `max_iter::Int`: Maximum number of iterations for the LP optimizer (default is 5000)
"""
function GenerativeModelProtocols.earth_mover_distance(
    protocol::GenerativeModelProtocols.GenerativeModelProtocol, 
    target_data::AbstractMatrix; 
    metric = Distances.Euclidean(),
    max_iter::Int = 5000)

    num_samples = size(target_data, 2)
    synthetic_data = protocol(num_samples)

    return GenerativeModelProtocols.earth_mover_distance(synthetic_data, target_data;
        metric   = metric,
        max_iter = max_iter
    )
end

"""
    GenerativeModelProtocols.earth_mover_distance(
        protocol::GenerativeModelProtocol; 
        metric = Distances.Euclidean(), max_iter::Int = 5000)

Calculates the Earth Mover's Distance between synthetic data produced by the 
protocol and its training data.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
 - `max_iter::Int`: Maximum number of iterations for the LP optimizer (default is 5000)
"""
function GenerativeModelProtocols.earth_mover_distance(
    protocol::GenerativeModelProtocols.GenerativeModelProtocol; 
    metric = Distances.Euclidean(), max_iter::Int = 5000)

    return GenerativeModelProtocols.earth_mover_distance(protocol, protocol.training_data; 
        metric   = metric,
        max_iter = max_iter
    )
end

end

