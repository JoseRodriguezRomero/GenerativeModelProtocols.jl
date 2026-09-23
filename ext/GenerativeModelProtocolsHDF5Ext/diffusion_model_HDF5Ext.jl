function GenerativeModelProtocols.load_diffusion_model_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String};
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name,
    tabular_denoiser_group_name::String = GenerativeModelProtocols.@default_tabular_denoiser_group_name)

    denoiser_model = GenerativeModelProtocols.load_tabular_denoiser_parameters(file;
        main_group_name             = main_group_name,
        generative_model_group_name = generative_model_group_name,
        tabular_denoiser_group_name = tabular_denoiser_group_name
    )

    HDF5.h5open(file.filename, "r") do file
        diffusion_model_parameters_group = file[main_group_name][generative_model_group_name]
        β = read(diffusion_model_parameters_group["beta"])

        return GenerativeModelProtocols.DiffusionModel(Tuple(β), denoiser_model)
    end
end

function GenerativeModelProtocols._save_model(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, 
    model::GenerativeModelProtocols.DiffusionModel;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name,
    tabular_denoiser_group_name::String = GenerativeModelProtocols.@default_tabular_denoiser_group_name)
    
    HDF5.h5open(file.filename, "w") do file
        main_group = HDF5.create_group(file, main_group_name)
        generative_model_group = HDF5.create_group(main_group, generative_model_group_name)

        HDF5.write(generative_model_group, "beta", collect(model.β))
    end

    GenerativeModelProtocols._save_model(file, model.denoiser_model;
        main_group_name             = main_group_name,
        generative_model_group_name = generative_model_group_name,
        tabular_denoiser_group_name = tabular_denoiser_group_name
    )
end

