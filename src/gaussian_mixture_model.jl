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

function compatible_gmm_model(k::Int, log_σ²::Matrix{Float64}, μ::Matrix{Float64}, predictor_network::Chain, p::Vector{Float64})
    if size(log_σ², 1) != k
        return false
    end

    if size(μ, 1) != k
        return false
    end

    if length(p) != k
        return false
    end

    if size(log_σ², 2) != size(μ, 2)
        return false
    end

    if (input_size(predictor_network) != size(log_σ², 2))
        return false
    end

    if (output_size(predictor_network) != k)
        return false
    end

    return true
end

@compat public GaussianMixtureModel

"""
$TYPEDEF

A structure containing the general parameters needed to evaluate and train a 
Gaussian Mixture Model (GMM). Once trained, it can be used a generative model.

$TYPEDFIELDS
"""
@kwdef struct GaussianMixtureModel <: AbstractCategoricalGenerativeModel
    """Number Gaussian clusters in the mixture model."""
    k::Int
    """Logarithm of the square of the variance of the Gaussian clusters of the model."""
    log_σ²::Matrix{Float64}
    """Mean values of the Gaussian clusters of the model."""
    μ::Matrix{Float64}
    """Neural network parametrizing the likelihood of some input stemming from a given cluster."""
    predictor_network::Chain
    """Vector containing the categorical probabilities of each cluster."""
    p::Vector{Float64}

    function GaussianMixtureModel(k::Int, log_σ²::Matrix{Float64}, μ::Matrix{Float64}, predictor_network::Chain, p::Vector{Float64})
        if !compatible_gmm_model(k, log_σ², μ, predictor_network, p)
            @error "Incompatible GaussianMixtureModel architecture!"
            throw(MethodError(GaussianMixtureModel, (k, log_σ², μ, predictor_network, p)))
        end

        return new(k, log_σ², μ, predictor_network, p)
    end
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
        predictor_network   = _gmm_default_default_predictor_network(input_size,k),
        p                   = ones(Float64,k) ./ Float64(k)
    )
end

function load_gaussian_mixture_parameters end

macro load_gaussian_mixture_parameters(saved_model, main_group_name, generative_model_group_name)
    return :(load_gaussian_mixture_parameters($(esc(saved_model));
        $(main_group_name = esc(main_group_name)),
        $(generative_model_group_name = esc(generative_model_group_name))
    ))
end

function GaussianMixtureModel(saved_model::String; 
    main_group_name = @default_main_group_name,
    generative_model_group_name = @default_generative_model_group_name)

    return @load_gaussian_mixture_parameters(saved_model, main_group_name, generative_model_group_name)
end

function Base.display(model::GaussianMixtureModel)
    print_padding = @_default_print_padding
    println("GenerativeModelProtocols.GaussianMixtureModel:")
    println("k      = $(model.k)")
    println("μ      = $(summary(model.μ))")
    println("log_σ² = $(summary(model.log_σ²))")
    println("p      = $(summary(model.p))")
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

function _generative_model(::GaussianMixtureModel)::GenerativeModel
    return gaussian_mixture_model
end

function log_gaussian_pdf_matrix(X::AbstractMatrix{Float64}, μ::AbstractMatrix{Float64}, log_σ²::AbstractMatrix{Float64})
    D = size(X, 1)
    K = size(μ, 1)
    
    X_3d = reshape(X, D, 1, :)
    μ_3d = reshape(transpose(μ), D, K, 1)
    inv_σ²_3d = reshape(transpose(exp.(-log_σ²)), D, K, 1)
    log_det_3d = reshape(sum(log_σ², dims=2), 1, K, 1) 
    
    diff = X_3d .- μ_3d                                
    mahalanobis = sum((diff .^ 2) .* inv_σ²_3d, dims=1) 
    
    log_P = dropdims(-0.5 .* (D * log(2π) .+ log_det_3d .+ mahalanobis), dims=1)
    
    return log_P
end

function _train!(protocol::GenerativeModelProtocol, model::GaussianMixtureModel; print_log::Bool = true)
    model_train_device = model |> protocol.device
    opt_state = Flux.setup(protocol.optimiser, model_train_device.predictor_network)
    batchsize_device = protocol.batchsize |> protocol.device
    training_data_device = Float64.(protocol.training_data) |> protocol.device
    shuffle_device = protocol.shuffle |> protocol.device
    
    K = model_train_device.k

    loader = load_data(training_data_device, batchsize_device, shuffle_device)

    if print_log; println("Training GMM via Global EM...") end
    for epoch in 1:protocol.epochs
        epoch_loss = 0.0
        total_grad_norm = 0.0

        π_network_all = softmax(model_train_device.predictor_network(training_data_device), dims=1) 
        log_P_all = log_gaussian_pdf_matrix(training_data_device, model_train_device.μ, model_train_device.log_σ²) 
        
        log_joint_all = log.(π_network_all .+ 1.0E-8) .+ log_P_all 
        
        max_log = maximum(log_joint_all, dims=1)          
        sum_exp = sum(exp.(log_joint_all .- max_log), dims=1) 
        log_total = max_log .+ log.(sum_exp)          
        
        γ_all = exp.(log_joint_all .- log_total)
        epoch_loss = -sum(log_total)

        N_k = sum(γ_all, dims=2) 
        N_k_stable = N_k .+ 1.0E-8
        
        model_train_device.μ .= (γ_all * transpose(training_data_device)) ./ N_k_stable
        
        X_3d = reshape(training_data_device, size(training_data_device, 1), 1, :)
        μ_3d = reshape(transpose(model_train_device.μ), size(training_data_device, 1), K, 1)
        γ_3d = reshape(γ_all, 1, K, :)
        
        diff_sq = (X_3d .- μ_3d) .^ 2 
        variance_matrix = dropdims(sum(γ_3d .* diff_sq, dims=3), dims=3) ./ transpose(N_k_stable)
        
        variance_matrix .= max.(variance_matrix, 0.0025)
        model_train_device.log_σ² .= transpose(log.(variance_matrix))

        if epoch == 1 || (epoch % 100 == 0 && protocol.shuffle)
            loader = load_data((training_data_device,γ_all), batchsize_device, shuffle_device)
        end

        for (x_batch, γ_batch) in loader
            _, grads = Flux.withgradient(model_train_device.predictor_network) do net
                pred = net(x_batch)
                Flux.Losses.logitcrossentropy(pred, γ_batch)
            end

            raw_gradient_arrays = Optimisers.trainables(grads)
            batch_grad_norm = sqrt(sum(sum(abs2, g) for g in raw_gradient_arrays if g isa AbstractArray))

            Flux.update!(opt_state, model_train_device.predictor_network, grads[1])
            total_grad_norm += batch_grad_norm
        end

        protocol._log.loss[epoch] = epoch_loss
        protocol._log.loss_grad_norm[epoch] = total_grad_norm / length(loader)

        if epoch % 5 == 0 || epoch == 1
            average_loss = protocol._log.loss[epoch]
            average_grad_norm = protocol._log.loss_grad_norm[epoch]
            if print_log
                @printf("Epoch %8d | Negative Log-Likelihood: %16.8e | Grad Norm: %16.8e \n", epoch, average_loss, average_grad_norm)
            end
        end
    end
    if print_log; println("Training complete!") end

    Flux.loadmodel!(model.μ, model_train_device.μ)
    Flux.loadmodel!(model.log_σ², model_train_device.log_σ²)
    Flux.loadmodel!(model.predictor_network, model_train_device.predictor_network)

    p = model.predictor_network(protocol.training_data)
    p = mean(softmax(transpose(p)),dims=1)
    model.p[:] = p[:]

    return protocol._log
end

function (model::GaussianMixtureModel)(category::Int, n_samples::Int)
    D = size(model.μ, 2)
    synthetic_X = randn(Float64, D, n_samples)
    σ = exp.(0.5 .* model.log_σ²)
    
    for i in 1:n_samples
        for d in 1:D
            synthetic_X[d, i] = model.μ[category, d] + σ[category, d] * synthetic_X[d, i]
        end
    end
    
    return synthetic_X
end

function (model::GaussianMixtureModel)(n_samples::Int)
    D = size(model.μ, 2)
    
    cum_p = cumsum(model.p)
    cum_p ./= cum_p[end]
    sampled_clusters = [searchsortedfirst(cum_p, rand()) for _ in 1:n_samples]
    
    counts = zeros(Int, model.k)
    for k in sampled_clusters
        counts[k] += 1
    end
    
    ends = cumsum(counts)
    starts = [1; ends[1:end-1] .+ 1]
    final_X = Matrix{Float64}(undef, D, n_samples)
    
    for k in 1:model.k
        counts[k] == 0 && continue
        
        final_X[:, starts[k]:ends[k]] .= model(k, counts[k])
    end
    
    return final_X
end

function (model::GaussianMixtureModel)()
    return model(1)[:]
end

"""
    categorize(model::GenerativeModelProtocols.GaussianMixtureModel, x::Vector) -> Vector

Compute the posterior probability distribution over the mixture components for a 
given input vector `x`. Returns a vector where the k-th element represents the 
conditional probability that the input stems from the k-th categorical cluster 
of the model.
"""
function categorize(model::GaussianMixtureModel, x::Matrix)::Matrix
    return model.predictor_network(x)
end

"""
    categorize(model::GenerativeModelProtocols.GaussianMixtureModel, x::Matrix) -> Matrix

Batch compute the posterior probability distributions over the mixture 
components for multiple input vectors. Each sample in the input matrix `x` is 
mapped to a normalized categorical probability vector where the k-th element 
represents the conditional probability that the sample stems from the k-th 
cluster.
"""
function categorize(model::GaussianMixtureModel, x::Vector)::Vector
    return model.predictor_network(x)
end

