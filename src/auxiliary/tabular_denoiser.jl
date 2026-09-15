macro tabular_denoiser_default_max_period()
    return 10000.0
end

function compatible_tabular_denoiser(
    T::Int, 
    time_embedding_mlp::Chain, 
    input_projection::Dense, 
    residual_layers::NamedTuple{LayerNames, <:Tuple{Vararg{Dense}}},
    time_projection_layers::NamedTuple{LayerNames, <:Tuple{Vararg{Dense}}}, 
    output_projection::Dense,
    max_period::F
    ) where {LayerNames, F<:AbstractFloat}

    if max_period ≤ 0
        return false
    end
    
    if _input_size(time_embedding_mlp) != T
        return false
    end

    if _input_size(input_projection) != _output_size(output_projection)
        return false
    end

    if _output_size(input_projection) != _input_size(output_projection)
        return false
    end

    if length(residual_layers) != length(time_projection_layers)
        return false
    end

    hidden_layer_size = _output_size(input_projection)

    function check_hidden_layers(layers)
        for layer in layers
            if _input_size(layer) != hidden_layer_size
                return false
            end
            
            if _output_size(layer) != hidden_layer_size
                return false
            end
        end

        return true
    end

    if !check_hidden_layers(residual_layers)
        return false
    end

    if !check_hidden_layers(time_projection_layers)
        return false
    end
    
    return true
end

@compat public TabularDenoiser

"""
$TYPEDEF

A time-conditioned residual neural network for vector-based diffusion 
models.latent_dim, latent_layers, encoders, decoders

$TYPEDFIELDS
"""
@kwdef struct TabularDenoiser{LayerNames, F<:AbstractFloat}
    """Total number of discrete time steps in the forward noising process and reverse denoising timeline."""
    T::Int
    """Multi-layer perceptron that maps static sinusoidal time frequencies into a globally shared learned temporal context vector."""
    time_embedding_mlp::Chain
    """Entry layer that projects the raw noisy input data vector into the initial hidden feature state."""
    input_projection::Dense
    """Tuple of sequential dense layers that transform hidden features inside the recurrent residual block loop."""
    residual_layers::NamedTuple{LayerNames, <:Tuple{Vararg{Dense}}}
    """Tuple of dense layers that map the shared temporal context into a dynamic additive bias shift inside the recurrent residual block loop."""
    time_projection_layers::NamedTuple{LayerNames, <:Tuple{Vararg{Dense}}}
    """Exit layer that projects final hidden features out of the residual block loop back to the original data dimensions to output the noise prediction."""
    output_projection::Dense
    """Constant that sets the maximum periodic scale for the base sinusoidal time step calculation."""
    max_period::F = F(@tabular_denoiser_default_max_period)
    """Trained parameters of the tabular denoiser. Users should not use directly use this."""
    _ps::Union{Ref{<:NamedTuple}, Nothing} = nothing
    """Trained state of the tabular denoiser. Users should not use directly use this."""
    _st::Union{Ref{<:NamedTuple}, Nothing} = nothing

    function TabularDenoiser(
    T::Int, 
    time_embedding_mlp::Chain, 
    input_projection::Dense, 
    residual_layers::NamedTuple{LayerNames, <:Tuple{Vararg{Dense}}},
    time_projection_layers::NamedTuple{LayerNames, <:Tuple{Vararg{Dense}}}, 
    output_projection::Dense,
    max_period::F,
    _ps::Union{Ref{<:NamedTuple}, Nothing},
    _st::Union{Ref{<:NamedTuple}, Nothing}
    ) where {LayerNames, F<:AbstractFloat}

        if !compatible_tabular_denoiser(T, time_embedding_mlp, input_projection, residual_layers, time_projection_layers, output_projection, max_period)
            @error "Incompatible TabularDenoiser architecture!"
            throw(MethodError(TabularDenoiser, (T, time_embedding_mlp, input_projection, residual_layers, time_projection_layers, output_projection, max_period)))
        end

        if isnothing(_ps) && isnothing(_st)
            _ps_val, _st_val = Lux.setup(Random.default_rng(), (
                time_embedding_mlp     = time_embedding_mlp, 
                input_projection       = input_projection, 
                residual_layers        = residual_layers,
                time_projection_layers = time_projection_layers,
                output_projection      = output_projection)
            )

            _ps = Ref{NamedTuple}(_ps_val)
            _st = Ref{NamedTuple}(_st_val)
        end

        return new{LayerNames, F}(T, time_embedding_mlp, input_projection, residual_layers, time_projection_layers, output_projection, max_period, _ps, _st)
    end
end

"""
    GenerativeModelProtocols.TabularDenoiser(num_inputs::Int; hidden_layer_size::Int = 64, T::Int = 32, activation_function::Function = relu, max_period::Float64 = 10000.0)

Convenience constructor to create a default `TabularDenoiser` that matches the
number of inputs specified by `num_inputs`.
"""
function TabularDenoiser(num_inputs::Int; 
    hidden_layer_size::Int = 64, 
    T::Int = 32, 
    activation_function::Function = swish,
    max_period::FP = @tabular_denoiser_default_max_period
    ) where {FP<:AbstractFloat}

    function dense(in_size::Int, out_size::Int, activation::Function = identity)
        return Dense(in_size => out_size, activation; init_weight=Lux.kaiming_uniform, init_bias=Lux.zeros32)
    end

    time_embedding_mlp = Chain(
        dense(T, hidden_layer_size, activation_function),
        dense(hidden_layer_size, hidden_layer_size, activation_function),
        dense(hidden_layer_size, hidden_layer_size, activation_function),
        dense(hidden_layer_size, hidden_layer_size, activation_function),
        dense(hidden_layer_size, hidden_layer_size),
    )

    input_projection = dense(num_inputs, hidden_layer_size, activation_function)

    residual_layers = (
        layer_1 = dense(hidden_layer_size, hidden_layer_size, activation_function),
        layer_2 = dense(hidden_layer_size, hidden_layer_size, activation_function),
        layer_3 = dense(hidden_layer_size, hidden_layer_size, activation_function)
    )

    time_projection_layers = (
        layer_1 = dense(hidden_layer_size, hidden_layer_size, activation_function),
        layer_2 = dense(hidden_layer_size, hidden_layer_size, activation_function),
        layer_3 = dense(hidden_layer_size, hidden_layer_size, activation_function)
    )

    output_projection = dense(hidden_layer_size, num_inputs)

    return TabularDenoiser(;
        T                      = T,
        time_embedding_mlp     = time_embedding_mlp,
        input_projection       = input_projection,
        residual_layers        = residual_layers,
        time_projection_layers = time_projection_layers,
        output_projection      = output_projection,
        max_period             = max_period
    )
end

function load_tabular_denoiser_parameters end

macro default_tabular_denoiser_group_name()
    return "tabular_denoiser"
end

macro load_tabular_denoiser_parameters(saved_model, main_group_name, generative_model_group_name, tabular_denoiser_group_name)
    return :(load_tabular_denoiser_parameters($(esc(saved_model)); 
        $(main_group_name = esc(main_group_name)), 
        $(generative_model_group_name = esc(generative_model_group_name)),
        $(tabular_denoiser_group_name = esc(tabular_denoiser_group_name))
    ))
end

function TabularDenoiser(saved_model::String; 
    main_group_name::String = @default_main_group_name,
    generative_model_group_name::String = @default_generative_model_group_name,
    tabular_denoiser_group_name::String = @default_tabular_denoiser_group_name)

    return @load_tabular_denoiser_parameters(saved_model, main_group_name, generative_model_group_name, tabular_denoiser_group_name)
end

function Base.display(denoiser_model::TabularDenoiser)
    println("$(Base.typename(typeof(denoiser_model)).wrapper):")
    println("T                 = $(denoiser_model.T)")
    println("max_period        = $(denoiser_model.max_period)")
    
    print("input_projection  = ")
    println(denoiser_model.input_projection)

    print("output_projection = ")
    println(denoiser_model.output_projection)
    println("")

    println("time_embedding_mlp:")
    _print_chains(denoiser_model.time_embedding_mlp)
    println("")

    println("residual_layers:")
    _print_layers(denoiser_model.residual_layers)
    println("")

    println("time_projection_layers:")
    _print_layers(denoiser_model.time_projection_layers)
    println("")
end

function compute_sinusoidal_frequencies(t, T::Int, max_period::AbstractFloat)
    half_dim = ceil(Int, T / 2.0)
    scale = log(max_period) / (half_dim - 1)
    
    frequencies = similar(t, eltype(t), half_dim)
    for i in 1:half_dim
        frequencies[i] = exp(-(i - 1) * scale)
    end
    
    scaled_time = t' .* frequencies
    sin_components = sin.(scaled_time)
    cos_components = cos.(scaled_time)
    
    return vcat(sin_components, cos_components)[1:T,:]
end

function _eval(m::TabularDenoiser, x, t, ps, st)
    static_time_features = compute_sinusoidal_frequencies(t, m.T, m.max_period)
    
    learned_time_context, new_time_st = m.time_embedding_mlp(static_time_features, ps.time_embedding_mlp, st.time_embedding_mlp)
    hidden_features, new_input_st = m.input_projection(x, ps.input_projection, st.input_projection)
    
    N = length(m.residual_layers)
    layer_keys = keys(m.residual_layers)

    function step_layer(i, current_hidden)
        feature_layer = m.residual_layers[i]
        time_layer = m.time_projection_layers[i]

        feature_layer_ps = ps.residual_layers[i]
        feature_layer_st = st.residual_layers[i]

        time_layer_ps = ps.time_projection_layers[i]
        time_layer_st = st.time_projection_layers[i]

        transformed_features, new_feat_st = feature_layer(current_hidden, feature_layer_ps, feature_layer_st)
        time_bias_shift, new_time_proj_st = time_layer(learned_time_context, time_layer_ps, time_layer_st)

        next_hidden = transformed_features .+ time_bias_shift .+ current_hidden
        return next_hidden, new_feat_st, new_time_proj_st
    end

    final_hidden = hidden_features

    outputs = ntuple(N) do i
        next_hidden, feat_st, time_st = step_layer(i, final_hidden)
        final_hidden = next_hidden
        return (feat_st, time_st)
    end

    out_features, new_out_st = m.output_projection(final_hidden, ps.output_projection, st.output_projection)

    updated_st = (
        time_embedding_mlp     = new_time_st,
        input_projection       = new_input_st,
        residual_layers        = NamedTuple{layer_keys}(ntuple(i -> outputs[i][1], N)),
        time_projection_layers = NamedTuple{layer_keys}(ntuple(i -> outputs[i][2], N)),
        output_projection      = new_out_st
    )
    
    return out_features, updated_st
end

function (m::TabularDenoiser)(x, t)
    return first(_eval(m, x, t, m._ps[], m._st[]))
end

