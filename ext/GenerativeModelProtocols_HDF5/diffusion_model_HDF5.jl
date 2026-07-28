function GenerativeModelProtocols.load_diffusion_model_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String};
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    HDF5.h5open(file.filename, "r") do file
        diffusion_model_parameters_group = file[main_group_name][generative_model_group_name]
        denoiser_group = diffusion_model_parameters_group["denoiser"]

        denoiser = GenerativeModelProtocols.ChainParameters(read_group_layer_parameters(denoiser_group))
        β = HDF5.attrs(diffusion_model_parameters_group)["beta"]
        T = HDF5.attrs(diffusion_model_parameters_group)["T"]

        return GenerativeModelProtocols.DiffusionModelParameters(T, β, denoiser)
    end
end

function GenerativeModelProtocols._save_model(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, 
    model::GenerativeModelProtocols.DiffusionModel;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    diffusion_model_parameters = GenerativeModelProtocols.diffusion_model_parameters(model)
    
    HDF5.h5open(file.filename, "w") do file
        main_group = HDF5.create_group(file, main_group_name)
        generative_model_group = HDF5.create_group(main_group, generative_model_group_name)
        denoiser_group = HDF5.create_group(generative_model_group, "denoiser")

        write_chain_to_file(diffusion_model_parameters.denoiser_model, denoiser_group)
        HDF5.write_attribute(generative_model_group, "beta", diffusion_model_parameters.β)
        HDF5.write_attribute(generative_model_group, "T", diffusion_model_parameters.T)
    end
end

