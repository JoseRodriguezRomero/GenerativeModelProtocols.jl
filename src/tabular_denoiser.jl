@compat public TabularDenoiser

"""
$TYPEDEF

A time-conditioned residual neural network for vector-based diffusion 
models.

$TYPEDFIELDS
"""
struct TabularDenoiser
    T::Int
    time_embedding_mlp::Chain
    input_projection::Dense
    residual_layers::Tuple{Vararg{Dense}}
    time_projection_layers::Tuple{Vararg{Dense}}
    output_projection::Dense
    max_period::Float64
end

Flux.@layer TabularDenoiser

"""
    GenerativeModelProtocols.TabularDenoiser(num_inputs::Int; hidden_layer_size::Int = 64, T::Int = 32, activation_function::Function = relu, max_period::Float64 = 10000.0)

Convenience constructor to create a default `TabularDenoiser` that matches the
number of inputs specified by `num_inputs`.
"""
function TabularDenoiser(num_inputs::Int; 
    hidden_layer_size::Int = 64, 
    T::Int = 32, 
    activation_function::Function = relu,
    max_period::Float64 = 10000.0)

    time_embedding_mlp = Chain(
        Dense(T => hidden_layer_size, activation_function), 
        Dense(hidden_layer_size => hidden_layer_size)
    )
    
    input_projection = Dense(num_inputs => hidden_layer_size, activation_function)
    
    residual_layers = (
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function)
    )
    
    time_projection_layers = (
        Dense(hidden_layer_size => hidden_layer_size),
        Dense(hidden_layer_size => hidden_layer_size),
        Dense(hidden_layer_size => hidden_layer_size)
    )
    
    output_projection = Dense(hidden_layer_size => num_inputs)
    
    return TabularDenoiser(
        T,
        time_embedding_mlp, 
        input_projection, 
        residual_layers, 
        time_projection_layers, 
        output_projection, 
        max_period
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

    denoiser_model_parameters = @load_tabular_denoiser_parameters(saved_model, main_group_name, generative_model_group_name, tabular_denoiser_group_name)
    return tabular_denoiser_parameters(denoiser_model_parameters)
end

function Base.display(denoiser_model::TabularDenoiser)
    print_padding = @_default_print_padding
    println("$(summary(denoiser_model)):")
    println("T                 = $(denoiser_model.T)")
    println("max_period        = $(denoiser_model.max_period)")
    
    print("input_projection  = ")
    println(denoiser_model.input_projection)

    print("output_projection = ")
    println(denoiser_model.output_projection)
    println("")

    println("time_embedding_mlp:")
    _print_chains(denoiser_model.time_embedding_mlp, print_padding)
    println("")

    println("residual_layers:")
    _print_layers(denoiser_model.residual_layers, print_padding)
    println("")

    println("time_projection_layers:")
    _print_layers(denoiser_model.time_projection_layers, print_padding)
    println("")
end

function compute_sinusoidal_frequencies(t, T::Int, max_period::Float64)
    half_dim = T ÷ 2
    
    scale = log(max_period) / (half_dim - 1)
    frequencies = exp.(collect(0:half_dim-1) * -scale)
    
    scaled_time = t' .* frequencies
    sin_components = sin.(scaled_time)
    cos_components = cos.(scaled_time)
    
    return vcat(sin_components, cos_components)
end

function (m::TabularDenoiser)(x, t)
    static_time_features = compute_sinusoidal_frequencies(t, m.T, m.max_period)
    learned_time_context = m.time_embedding_mlp(static_time_features)
    hidden_features = m.input_projection(x)
    
    for (feature_layer, time_layer) in zip(m.residual_layers, m.time_projection_layers)
        transformed_features = feature_layer(hidden_features)
        time_bias_shift = time_layer(learned_time_context)
        hidden_features = transformed_features .+ time_bias_shift .+ hidden_features
    end
    
    return m.output_projection(hidden_features)
end

