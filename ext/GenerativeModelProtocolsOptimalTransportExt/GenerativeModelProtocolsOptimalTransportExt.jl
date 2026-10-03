module GenerativeModelProtocolsOptimalTransportExt

import GenerativeModelProtocols

using Distances
using OptimalTransport

"""
    GenerativeModelProtocols.sinkhorn_distance(
        synthetic_data::AbstractMatrix, target_data::AbstractMatrix; 
        metric = Distances.Euclidean(), ε::Float64 = 0.05, 
        alg = SinkhornGibbs(), atol::Float64 = 0.0, 
        rtol::Float64 = atol > 0 ? 0 : √eps(Float64), 
        check_convergence::Int = 10, maxiter::Int = 100_000,
        regularization::Bool = false)    

Calculates the Sinkhorn distance between synthetic and target data.

# Arguments
 - `synthetic_data::AbstractMatrix`: A matrix of synthetic data points
 - `target_data::AbstractMatrix`: A matrix of target data points

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
 - `ε::Float64`: The regularization parameter for Sinkhorn divergence (default is 0.05)
 - `alg`: The algorithm to use for solving the Sinkhorn problem (default is SinkhornGibbs)
 - `atol::Float64`: Absolute tolerance for convergence (default is 0)
 - `rtol::Float64`: Relative tolerance for convergence (default is 0 if atol > 0, otherwise √eps(Float64))
 - `check_convergence::Int`: Number of iterations between convergence checks (default is 10)
 - `maxiter::Int`: Maximum number of iterations (default is 100_000)
 - `regularization::Bool`: Whether to add the regularization term to the returned distance (default is false)
"""
function GenerativeModelProtocols.sinkhorn_distance(
    synthetic_data::AbstractMatrix, target_data::AbstractMatrix; 
    metric = Distances.Euclidean(), ε::Float64 = 0.05, alg = SinkhornGibbs(),
    atol::Float64 = 0.0, rtol::Float64 = atol > 0 ? 0 : √eps(Float64), 
    check_convergence::Int = 10, maxiter::Int = 100_000, 
    regularization::Bool = false)

    num_samples = size(target_data, 2)
    μ = fill(1.0 / num_samples, num_samples)
    ν = fill(1.0 / num_samples, num_samples)
    C = GenerativeModelProtocols._pairwise_metric_distance(synthetic_data, target_data, metric)
    
    return sinkhorn2(μ, ν, C, ε, alg; 
        atol              = atol,
        rtol              = rtol,
        check_convergence = check_convergence,
        maxiter           = maxiter,
        regularization    = regularization
    )
end

"""
    GenerativeModelProtocols.sinkhorn_distance(
        protocol::GenerativeModelProtocol, target_data::AbstractMatrix; 
        metric = Distances.Euclidean(), ε::Float64 = 0.05, 
        alg = SinkhornGibbs(), atol::Float64 = 0.0, 
        rtol::Float64 = atol > 0 ? 0 : √eps(Float64), 
        check_convergence::Int = 10, maxiter::Int = 100_000,
        regularization::Bool = false)

Calculates the Sinkhorn distance between synthetic data produced by the protocol 
and target data.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use
 - `target_data::AbstractMatrix`: A matrix of target data points

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
 - `ε::Float64`: The regularization parameter for Sinkhorn divergence (default is 0.05)
 - `alg`: The algorithm to use for solving the Sinkhorn problem (default is SinkhornGibbs)
 - `atol::Float64`: Absolute tolerance for convergence (default is 0)
 - `rtol::Float64`: Relative tolerance for convergence (default is 0 if atol > 0, otherwise √eps(Float64))
 - `check_convergence::Int`: Number of iterations between convergence checks (default is 10)
 - `maxiter::Int`: Maximum number of iterations (default is 100_000)
 - `regularization::Bool`: Whether to add the regularization term to the returned distance (default is false)
"""
function GenerativeModelProtocols.sinkhorn_distance(
    protocol::GenerativeModelProtocols.GenerativeModelProtocol, 
    target_data::AbstractMatrix; 
    metric = Distances.Euclidean(), ε::Float64 = 0.05, alg = SinkhornGibbs(),
    atol::Float64 = 0.0, rtol::Float64 = atol > 0 ? 0 : √eps(Float64), 
    check_convergence::Int = 10, maxiter::Int = 100_000, 
    regularization::Bool = false)

    num_samples = size(target_data, 2)
    synthetic_data = protocol(num_samples)
    return GenerativeModelProtocols.sinkhorn_distance(synthetic_data, target_data;
        metric            = metric,
        ε                 = ε, 
        alg               = alg,
        atol              = atol, 
        rtol              = rtol, 
        check_convergence = check_convergence, 
        maxiter           = maxiter,
        regularization    = regularization
    )
end


"""
    GenerativeModelProtocols.sinkhorn_distance(
        protocol::GenerativeModelProtocol; 
        metric = Distances.Euclidean(), ε::Float64 = 0.05, 
        alg = SinkhornGibbs(), atol::Float64 = 0.0, 
        rtol::Float64 = atol > 0 ? 0 : √eps(Float64), 
        check_convergence::Int = 10, maxiter::Int = 100_000,
        regularization::Bool = false)

Calculates the Sinkhorn distance between synthetic data produced by the protocol 
and its training data.

# Arguments
 - `protocol::GenerativeModelProtocol`: The generative model protocol to use

# Keyword Arguments
 - `metric::Metric`: The distance metric to use (default is Euclidean)
 - `ε::Float64`: The regularization parameter for Sinkhorn divergence (default is 0.05)
 - `alg`: The algorithm to use for solving the Sinkhorn problem (default is SinkhornGibbs)
 - `atol::Float64`: Absolute tolerance for convergence (default is 0)
 - `rtol::Float64`: Relative tolerance for convergence (default is 0 if atol > 0, otherwise √eps(Float64))
 - `check_convergence::Int`: Number of iterations between convergence checks (default is 10)
 - `maxiter::Int`: Maximum number of iterations (default is 100_000)
 - `regularization::Bool`: Whether to add the regularization term to the returned distance (default is false)
"""
function GenerativeModelProtocols.sinkhorn_distance(
    protocol::GenerativeModelProtocols.GenerativeModelProtocol; 
    metric = Distances.Euclidean(), ε::Float64 = 0.05, alg = SinkhornGibbs(),
    atol::Float64 = 0.0, rtol::Float64 = atol > 0 ? 0 : √eps(Float64), 
    check_convergence::Int = 10, maxiter::Int = 100_000, 
    regularization::Bool = false)

    return GenerativeModelProtocols.sinkhorn_distance(
        protocol, protocol.training_data;
        metric = metric,
        ε                 = ε,
        alg               = alg,
        atol              = atol,
        rtol              = rtol,
        check_convergence = check_convergence,
        maxiter           = maxiter,
        regularization    = regularization
    )
end

end

