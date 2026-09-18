module GenerativeModelProtocols_HDF5

using Lux

import GenerativeModelProtocols
import FileIO, HDF5

function make_enum_type(enum::Enum)
    hdf5_enum_types = HDF5.Datatype(HDF5.API.h5t_create(HDF5.API.H5T_ENUM, sizeof(enum)))
    enum_instances = instances(typeof(enum))

    for key in keys(enum_instances)
        raw_int_value = Integer(enum_instances[key])
        HDF5.API.h5t_enum_insert(hdf5_enum_types.id, string(enum_instances[key]), Ref(raw_int_value))
    end

    return hdf5_enum_types
end

function GenerativeModelProtocols._save_metadata(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, 
    protocol::GenerativeModelProtocols.GenerativeModelProtocol; 
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name, 
    metadata_group_name::String = GenerativeModelProtocols.@default_metadata_group_name,
    metadata::Dict{String,Any} = Dict{String,Any}())

    HDF5.h5open(file.filename, "r+") do file
        main_group = file[main_group_name]
        metadata_group = HDF5.create_group(main_group, metadata_group_name)
        scalar_space = HDF5.Dataspace(HDF5.API.h5s_create(HDF5.API.H5S_SCALAR))

        function write_attribute_enum(key::String, value)
            val_dt = make_enum_type(value)
            val_attr = HDF5.create_attribute(metadata_group, key, val_dt, scalar_space)
            HDF5.write_attribute(val_attr, val_dt, Integer(value))
        end

        write_attribute_enum("generative_model", GenerativeModelProtocols._generative_model(protocol))
        HDF5.write(metadata_group, "var_training_data", collect(protocol.var_training_data))
        HDF5.write(metadata_group, "mean_training_data", collect(protocol.mean_training_data))

        for metadata_key in keys(metadata)
            if typeof(metadata[metadata_key]) <: Enum
                write_attribute_enum(metadata_key, metadata[metadata_key])
            else
                HDF5.write_attribute(metadata_group, metadata_key, metadata[metadata_key])
            end
        end
    end
end

function GenerativeModelProtocols._read_metadata(file::FileIO.File{FileIO.DataFormat{:HDF5},String}; 
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name, 
    metadata_group_name::String = GenerativeModelProtocols.@default_metadata_group_name)

    HDF5.h5open(file.filename, "r") do file
        main_group = file[main_group_name]
        metadata_group = main_group[metadata_group_name]

        generative_model_map = GenerativeModelProtocols.generative_model_map()
        generative_model = generative_model_map[UInt8(read(HDF5.attributes(metadata_group)["generative_model"]))]
        mean_training_data = Tuple(read(metadata_group["mean_training_data"]))
        var_training_data = Tuple(read(metadata_group["var_training_data"]))
        
        return generative_model, mean_training_data, var_training_data
    end
end

function write_layer_to_file(layer::Dense, ps::NamedTuple, layer_group)
    activation_function_map = GenerativeModelProtocols.activation_function_map()
    lay_dt = make_enum_type(instances(GenerativeModelProtocols.ActivationFunction)[1])
    scalar_space = HDF5.Dataspace(HDF5.API.h5s_create(HDF5.API.H5S_SCALAR))

    attr_lay = HDF5.create_attribute(layer_group, "activation_function", lay_dt, scalar_space)
    HDF5.write_attribute(attr_lay, lay_dt, Int64(activation_function_map[layer.activation]))
    HDF5.write(layer_group, "bias", ps.bias)
    HDF5.write(layer_group, "weight", ps.weight)
end

function write_layers_to_file(layers::NamedTuple{LayerNames, <:Tuple{Vararg{Dense}}}, ps::NamedTuple{LayerNames, <:Tuple}, layers_group) where {LayerNames}
    for i in eachindex(layers)
        layer = layers[i]
        layer_group = HDF5.create_group(layers_group, "$i")
        write_layer_to_file(layer, ps[i], layer_group)
    end
end

function write_chain_to_file(chain::Chain, ps::NamedTuple, chain_group)
    write_layers_to_file(chain.layers, ps, chain_group)
end

function read_group_layer_parameters(layer_group)
    activation_function_inverse_map = GenerativeModelProtocols.activation_function_inverse_map()

    W = read(layer_group["weight"])
    bias = read(layer_group["bias"])
    σ = activation_function_inverse_map[UInt8(read(HDF5.attributes(layer_group)["activation_function"]))]
    
    out_dims, in_dims = size(W) 
    layer = Dense(in_dims => out_dims, σ)
    
    ps = (weight = W, bias = bias)

    return layer, ps
end

function read_group_layers_parameters(layers_group)
    n_layers = length(layers_group)
    layer_names = ["layer_$i" for i in 1:n_layers]
    
    results = [read_group_layer_parameters(layers_group[name]) for name in layer_names]
    
    layer_keys = Tuple(Symbol(name) for name in layer_names)
    layers_nt = NamedTuple{layer_keys}(Tuple(r[1] for r in results))
    ps_nt = NamedTuple{layer_keys}(Tuple(r[2] for r in results))
    
    return layers_nt, ps_nt
end

function read_group_chain_parameters(chain_group)
    layers_nt, ps_nt = read_group_layers_parameters(chain_group)

    return Chain(Tuple(layers_nt)), ps_nt
end

# Auxiliary scripts
include("tabular_denoiser_HDF5.jl")

# Generative models
include("diffusion_model_HDF5.jl")
include("normalizing_flow_HDF5.jl")
include("gaussian_mixture_model_HDF5.jl")
include("variational_autoencoder_HDF5.jl")
include("generative_adversarial_network_HDF5.jl")

end

