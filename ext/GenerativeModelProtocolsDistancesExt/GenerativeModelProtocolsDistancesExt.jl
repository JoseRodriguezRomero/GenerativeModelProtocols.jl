module GenerativeModelProtocolsDistancesExt

import GenerativeModelProtocols

using Distances

function GenerativeModelProtocols._pairwise_metric_distance( 
    synthetic_data::AbstractMatrix, target_data::AbstractMatrix, metric)
    return pairwise(metric, target_data, synthetic_data, dims = 2)
end

"""
    GenerativeModelProtocols.energy_distance(
        synthetic_data::AbstractMatrix, target_data::AbstractMatrix;
        metric = Distances.Euclidean())

Calculates the Energy Distance between synthetic and target data.

# Arguments
 - `synthetic_data::AbstractMatrix`: A matrix of synthetic data points
 - `target_data::AbstractMatrix`: A matrix of target data points

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
"""
function GenerativeModelProtocols.energy_distance(
    synthetic_data::AbstractMatrix, target_data::AbstractMatrix;
    metric = Distances.Euclidean())
    
    x_samples = size(target_data, 2)
    y_samples = size(synthetic_data, 2)
    
    xx_samples = x_samples * (x_samples - 1)
    yy_samples = y_samples * (y_samples - 1)
    xy_samples = x_samples * y_samples

    C_xx = GenerativeModelProtocols._pairwise_metric_distance(target_data, target_data, metric)
    C_yy = GenerativeModelProtocols._pairwise_metric_distance(synthetic_data, synthetic_data, metric)
    C_xy = GenerativeModelProtocols._pairwise_metric_distance(target_data, synthetic_data, metric)

    mean_xx = sum(C_xx) / xx_samples
    mean_yy = sum(C_yy) / yy_samples
    mean_xy = sum(C_xy) / xy_samples

    return 2*mean_xy - mean_xx - mean_yy
end

"""
    GenerativeModelProtocols.energy_distance(
        protocol::GenerativeModelProtocol, target_data::AbstractMatrix; 
        metric = Distances.Euclidean())

Calculates the Energy Distance between synthetic data produced by the protocol 
and target data.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use
 - `target_data::AbstractMatrix`: A matrix of target data points

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
"""
function GenerativeModelProtocols.energy_distance(
    protocol::GenerativeModelProtocols.GenerativeModelProtocol,
    target_data::AbstractMatrix;
    metric = Distances.Euclidean())
    
    num_samples = size(target_data, 2)
    synthetic_data = protocol(num_samples)

    return GenerativeModelProtocols.energy_distance(synthetic_data, target_data; 
        metric = metric
    )
end

"""
    GenerativeModelProtocols.energy_distance(
        protocol::GenerativeModelProtocol; 
        metric = Distances.Euclidean())

Calculates the Energy Distance between synthetic data produced by the protocol 
and its training data.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
"""
function GenerativeModelProtocols.energy_distance(
    protocol::GenerativeModelProtocols.GenerativeModelProtocol; 
    metric = Distances.Euclidean())

    return GenerativeModelProtocols.energy_distance(protocol, protocol.training_data; 
        metric = metric
    )
end

end

