macro _gmm_default_activation_function()
    return swish
end

function _gmm_default_predictor_network(input_size::Int, k::Int, hidden_layer_size::Int = 32, activation_function::Function = @_gmm_default_activation_function)
    return Chain(
        Dense(input_size => hidden_layer_size, activation_function; init_weight=Lux.kaiming_uniform, init_bias=Lux.zeros32),
        Dense(hidden_layer_size => hidden_layer_size, activation_function; init_weight=Lux.kaiming_uniform, init_bias=Lux.zeros32),
        Dense(hidden_layer_size => hidden_layer_size, activation_function; init_weight=Lux.kaiming_uniform, init_bias=Lux.zeros32),
        Dense(hidden_layer_size => hidden_layer_size, activation_function; init_weight=Lux.kaiming_uniform, init_bias=Lux.zeros32),
        Dense(hidden_layer_size => hidden_layer_size, activation_function; init_weight=Lux.kaiming_uniform, init_bias=Lux.zeros32),
        Dense(hidden_layer_size => k)
    )
end

function compatible_gmm_model(log_σ²::Matrix{F}, μ::Matrix{F}, predictor_network::Chain, p::Vector{F}) where {F<:AbstractFloat}
    if size(μ, 1) != size(log_σ², 1)
        return false
    end

    if length(p) != size(log_σ², 1)
        return false
    end

    if size(log_σ², 2) != size(μ, 2)
        return false
    end

    if (_input_size(predictor_network) != size(log_σ², 2))
        return false
    end

    if (_output_size(predictor_network) != size(log_σ², 1))
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
@kwdef struct GaussianMixtureModel{F<:AbstractFloat} <: AbstractCategoricalGenerativeModel
    """Logarithm of the square of the variance of the Gaussian clusters of the model."""
    log_σ²::Matrix{F}
    """Mean values of the Gaussian clusters of the model."""
    μ::Matrix{F}
    """Neural network parametrizing the likelihood of some input stemming from a given cluster."""
    predictor_network::Chain
    """Vector containing the categorical probabilities of each cluster."""
    p::Vector{F}
    """Trained parameters of the predictor model. Users should not use directly use this."""
    _ps::Union{Ref{<:NamedTuple}, Nothing} = nothing
    """Trained state of the predictor model. Users should not use directly use this."""
    _st::Union{Ref{<:NamedTuple}, Nothing} = nothing

    function GaussianMixtureModel(
        log_σ²::Matrix{F}, 
        μ::Matrix{F}, 
        predictor_network::Chain, 
        p::Vector{F},
        _ps::Union{Ref{<:NamedTuple}, Nothing},
        _st::Union{Ref{<:NamedTuple}, Nothing}
        ) where {F<:AbstractFloat}

        if !compatible_gmm_model(log_σ², μ, predictor_network, p)
            @error "Incompatible GaussianMixtureModel architecture!"
            throw(MethodError(GaussianMixtureModel, (log_σ², μ, predictor_network, p)))
        end

        if isnothing(_ps) && isnothing(_st)
            _ps_val, _st_val = Lux.setup(Random.default_rng(), (predictor_network = predictor_network,))
            _ps = Ref{NamedTuple}(_ps_val)
            _st = Ref{NamedTuple}(_st_val)
        end

        return new{F}(log_σ², μ, predictor_network, p, _ps, _st)
    end
end

"""
    GenerativeModelProtocols.GaussianMixtureModel(input_size::Int, k::Int)

Convenience constructor that creates a 
`GenerativeModelProtocols.GaussianMixtureModel` using default predictor network 
architecture.
"""
function GaussianMixtureModel(input_size::Int, k::Int)
    return GaussianMixtureModel(;
        log_σ²              = zeros(Float64,k,input_size),
        μ                   = zeros(Float64,k,input_size),
        predictor_network   = _gmm_default_predictor_network(input_size,k),
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
    println("$(summary(model)):")
    println("μ      = $(summary(model.μ))")
    println("log_σ² = $(summary(model.log_σ²))")
    println("p      = $(summary(model.p))")
    println("")

    println("predictor_network: ")
    _print_chains(model.predictor_network, print_padding)
end

function _input_size(model::GaussianMixtureModel)::Int
    return size(model.log_σ², 2)
end

function _latent_size(model::GaussianMixtureModel)::Int
    return size(model.log_σ², 1)
end 

function _generative_model(::GaussianMixtureModel)::GenerativeModel
    return gaussian_mixture_model
end

function log_gaussian_pdf_matrix(X, μ, log_σ²)
    T = eltype(μ)
    D = size(X, 1)
    K = size(μ, 1)
    
    X_3d = reshape(X, D, 1, :)
    μ_3d = reshape(transpose(μ), D, K, 1)
    inv_σ²_3d = reshape(transpose(exp.(-log_σ²)), D, K, 1)
    log_det_3d = reshape(sum(log_σ², dims=2), 1, K, 1) 
    
    diff = X_3d .- μ_3d                                
    mahalanobis = sum((diff .^ 2) .* inv_σ²_3d, dims=1) 
    
    log_P = dropdims(-T(0.5) .* (D * T(log(2π)) .+ log_det_3d .+ mahalanobis), dims=1)
    
    return log_P
end

function _train!(protocol::GenerativeModelProtocol, model::GaussianMixtureModel; print_log::Bool = true)
    batchsize_device = protocol.batchsize |> protocol.device
    training_data_device = protocol.training_data |> protocol.device
    shuffle_device = protocol.shuffle |> protocol.device

    model_μ_device = model.μ |> protocol.device
    model_log_σ²_device = model.log_σ² |> protocol.device
    
    T = eltype(protocol.training_data)
    K = _latent_size(model) |> protocol.device

    predictor = model.predictor_network
    ps = protocol.precision(model._ps[]) |> protocol.device
    st = protocol.precision(model._st[]) |> protocol.device

    protocol._log["Mean Log-Likelihood"] = zeros(T, protocol.epochs)

    function _compute_gaussian_kernels(μ, log_σ²)
        pred_out, _ = predictor(training_data_device, ps.predictor_network, st.predictor_network)
        π_network_all = softmax(pred_out, dims=1) 
        log_P_all = log_gaussian_pdf_matrix(training_data_device, μ, log_σ²)
        
        log_joint_all = log.(π_network_all .+ T(1.0E-8)) .+ log_P_all 
        
        max_log = maximum(log_joint_all, dims=1)          
        sum_exp = sum(exp.(log_joint_all .- max_log), dims=1) 
        log_total = max_log .+ log.(sum_exp)          
        
        γ_all = exp.(log_joint_all .- log_total)
        epoch_loss = -mean(log_total)

        N_k = sum(γ_all, dims=2) 
        N_k_stable = N_k .+ T(1.0E-8)
        
        next_μ = (γ_all * transpose(training_data_device)) ./ N_k_stable
        
        X_3d = reshape(training_data_device, size(training_data_device, 1), 1, :)
        μ_3d = reshape(transpose(next_μ), size(training_data_device, 1), K, 1)
        γ_3d = reshape(γ_all, 1, K, :)
        
        diff_sq = (X_3d .- μ_3d) .^ 2 
        variance_matrix = dropdims(sum(γ_3d .* diff_sq, dims=3), dims=3) ./ transpose(N_k_stable)
        
        variance_matrix .= max.(variance_matrix, T(0.0025))
        next_log_σ² = identity.(transpose(log.(variance_matrix)))

        return epoch_loss, γ_all, next_μ, next_log_σ²
    end

    function _predictor_train_step!(data_batch, p_current, s_current, o_current)
        x_batch = data_batch[1]
        γ_batch = data_batch[2]

        logitcrossentropy = CrossEntropyLoss(; logits=Val(true))

        function _objective(predictor, p, s)
            pred = first(predictor(x_batch, p.predictor_network, s.predictor_network))
            return logitcrossentropy(pred, γ_batch)
        end
        
        loss_val = _objective(predictor, p_current, s_current)
        loss_grads = Enzyme.make_zero(p_current)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(_objective),
            Enzyme.Active,
            Enzyme.Const(predictor),
            Enzyme.Duplicated(p_current, loss_grads),
            Enzyme.Const(s_current)
        )

        o_updated, p_updated = Optimisers.update(o_current, p_current, loss_grads)
        return loss_val, p_updated, o_updated
    end

    compute_gaussian_kernels = _function_device_dispatch(protocol.device, _compute_gaussian_kernels, model_μ_device, model_log_σ²_device)
    _, γ_all, model_μ_device, model_log_σ²_device = compute_gaussian_kernels(model_μ_device, model_log_σ²_device)

    opt_state = _initial_step(model.predictor_network, ps, st, protocol.optimiser)
    loader = load_data((training_data_device, γ_all), batchsize_device, shuffle_device)

    _train_step!, opt_state = _train_step_device_dispatch(protocol.device, _predictor_train_step!, loader, opt_state)

    if print_log; println("Training GMM via Global EM...") end
    for epoch in 1:protocol.epochs
        epoch_loss, γ_all, model_μ_device, model_log_σ²_device = compute_gaussian_kernels(model_μ_device, model_log_σ²_device)

        if epoch == 1 || (epoch % 100 == 0 && protocol.shuffle)
            loader = load_data((training_data_device, γ_all), batchsize_device, shuffle_device)
        end

        for (x_batch, γ_batch) in loader
            _, opt_state = _train_step!((x_batch, γ_batch), opt_state)
        end

        mean_loss = T(-1.0) * epoch_loss
        protocol._log["Mean Log-Likelihood"][epoch] = mean_loss

        if print_log && (epoch % 5 == 0 || epoch == 1)
            @printf("Epoch %8d | Mean Log-Likelihood: %16.8e \n", epoch, mean_loss)
        end
    end
    if print_log; println("Training complete!") end

    function _load_model!(dst, src)
        dst[:] = (src |> cpu_device())[:]
    end

    _load_model!(model.μ, model_μ_device)
    _load_model!(model.log_σ², model_log_σ²_device)

    model._ps[] = opt_state.parameters
    model._st[] = opt_state.states

    p = first(model.predictor_network(protocol.training_data, model._ps[].predictor_network, model._st[].predictor_network))
    p = mean(softmax(transpose(p)),dims=1)
    model.p[:] = p[:]

    return protocol._log
end

function _eval(model::GaussianMixtureModel, category::Int, n_samples::Int)
    T = eltype(model.μ)
    D = size(model.μ, 2)
    synthetic_X = randn(Float64, D, n_samples)
    σ = exp.(T(0.5) .* model.log_σ²)
    
    for i in 1:n_samples
        for d in 1:D
            synthetic_X[d, i] = model.μ[category, d] + σ[category, d] * synthetic_X[d, i]
        end
    end
    
    return synthetic_X
end

function _eval(model::GaussianMixtureModel, n_samples::Int)
    D = size(model.μ, 2)
    
    cum_p = cumsum(model.p)
    cum_p ./= cum_p[end]
    sampled_clusters = [searchsortedfirst(cum_p, rand()) for _ in 1:n_samples]
    
    k = _latent_size(model)

    counts = zeros(Int, k)
    for k in sampled_clusters
        counts[k] += 1
    end
    
    ends = cumsum(counts)
    starts = [1; ends[1:end-1] .+ 1]
    final_X = Matrix{Float64}(undef, D, n_samples)
    final_X = similar(model.μ, eltype(model.μ), D, n_samples)
    
    for k in 1:k
        counts[k] == 0 && continue
        
        final_X[:, starts[k]:ends[k]] .= _eval(model, k, counts[k])
    end
    
    return final_X
end

function _eval(model::GaussianMixtureModel)
    return _eval(model, 1)[:]
end

function _categorize(model::GaussianMixtureModel, x::Matrix)::Matrix 
    return first(model.predictor_network(x, model._ps[].predictor_network, model._st[].predictor_network))
end

function _categorize(model::GaussianMixtureModel, x::Vector)::Vector 
    return first(model.predictor_network(x, model._ps[].predictor_network, model._st[].predictor_network))
end

