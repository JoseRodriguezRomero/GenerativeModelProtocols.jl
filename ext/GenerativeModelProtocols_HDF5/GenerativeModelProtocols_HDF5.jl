module GenerativeModelProtocols_HDF5

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

        for metadata_key in keys(metadata)
            if typeof(metadata[metadata_key]) <: Enum
                write_attribute_enum(metadata_key, metadata[metadata_key])
            else
                HDF5.write_attribute(metadata_group, metadata_key, metadata[metadata_key])
            end
        end
    end
end

function write_chain_to_file(chain, chain_group)
    activation_function_map = GenerativeModelProtocols.enum_activation_function_map()
    lay_dt = make_enum_type(instances(GenerativeModelProtocols.ActivationFunction)[1])
    scalar_space = HDF5.Dataspace(HDF5.API.h5s_create(HDF5.API.H5S_SCALAR))

    for i in eachindex(chain.layers)
        layer = chain.layers[i]
        layer_group = HDF5.create_group(chain_group, "layer $i")
        attr_lay = HDF5.create_attribute(layer_group, "activation_function", lay_dt, scalar_space)
        HDF5.write_attribute(attr_lay, lay_dt, Int64(activation_function_map[layer.activation_function]))
        HDF5.write(layer_group, "node_bias", layer.node_bias)
        HDF5.write(layer_group, "node_weights", layer.node_weights)
    end
end

function read_group_layer_parameters(chain_group)
    activation_function_inverse_map = GenerativeModelProtocols.enum_activation_function_inverse_map()
    layers = Vector{GenerativeModelProtocols.LayerParameters}(undef, length(keys(chain_group)))
    for i in eachindex(layers)
        layer_group = chain_group["layer $i"]
        new_layer = GenerativeModelProtocols.LayerParameters(
            read(layer_group["node_bias"]),
            read(layer_group["node_weights"]),
            activation_function_inverse_map[UInt8(read(HDF5.attributes(layer_group)["activation_function"]))]
        )
        layers[i] = new_layer
    end

    return layers
end

include("GenerativeModelVAE_HDF5.jl")
include("GenerativeModelDM_HDF5.jl")

end

