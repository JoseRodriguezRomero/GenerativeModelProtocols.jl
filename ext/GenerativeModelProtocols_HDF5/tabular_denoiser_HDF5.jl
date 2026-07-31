function GenerativeModelProtocols.load_tabular_denoiser_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String};
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name,
    tabular_denoiser_group_name::String = GenerativeModelProtocols.@default_tabular_denoiser_group_name)
    
    HDF5.h5open(file.filename, "r") do file
        generative_model_parameters_group = file[main_group_name][generative_model_group_name]
        tabular_denoiser_group = generative_model_parameters_group[tabular_denoiser_group_name]
        time_embedding_mlp_group = tabular_denoiser_group["time_embedding_mlp"]
        input_projection_group = tabular_denoiser_group["input_projection"]
        residual_layers_group = tabular_denoiser_group["residual_layers"]
        time_projection_layers_group = tabular_denoiser_group["time_projection_layers"]
        output_projection_group = tabular_denoiser_group["output_projection"]

        T = HDF5.attrs(tabular_denoiser_group)["T"]
        time_embedding_mlp = read_group_chain_parameters(time_embedding_mlp_group)
        input_projection = read_group_layer_parameters(input_projection_group)
        residual_layers = read_group_layers_parameters(residual_layers_group)
        time_projection_layers = read_group_layers_parameters(time_projection_layers_group)
        output_projection = read_group_layer_parameters(output_projection_group)
        max_period = HDF5.attrs(tabular_denoiser_group)["max_period"]

        return GenerativeModelProtocols.TabularDenoiser(;
            T                       = T,
            time_embedding_mlp      = time_embedding_mlp,
            input_projection        = input_projection,
            residual_layers         = residual_layers,
            time_projection_layers  = time_projection_layers,
            output_projection       = output_projection,
            max_period              = max_period
        )
    end
end

function GenerativeModelProtocols._save_model(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, 
    model::GenerativeModelProtocols.TabularDenoiser;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name,
    tabular_denoiser_group_name::String = GenerativeModelProtocols.@default_tabular_denoiser_group_name)

    HDF5.h5open(file.filename, "r+") do file
        main_group = file[main_group_name]
        generative_model_group = main_group[generative_model_group_name]
        tabular_denoiser_group = HDF5.create_group(generative_model_group, tabular_denoiser_group_name)
        
        time_embedding_mlp_group = HDF5.create_group(tabular_denoiser_group, "time_embedding_mlp")
        input_projection_group = HDF5.create_group(tabular_denoiser_group, "input_projection")
        residual_layers_group = HDF5.create_group(tabular_denoiser_group, "residual_layers")
        time_projection_layers_group = HDF5.create_group(tabular_denoiser_group, "time_projection_layers")
        output_projection_group = HDF5.create_group(tabular_denoiser_group, "output_projection")

        HDF5.write_attribute(tabular_denoiser_group, "T", model.T)
        write_chain_to_file(model.time_embedding_mlp, time_embedding_mlp_group)
        write_layer_to_file(model.input_projection, input_projection_group)
        write_layers_to_file(model.residual_layers, residual_layers_group)
        write_layers_to_file(model.time_projection_layers, time_projection_layers_group)
        write_layer_to_file(model.output_projection, output_projection_group)
        HDF5.write_attribute(tabular_denoiser_group, "max_period", model.max_period)
    end
end

