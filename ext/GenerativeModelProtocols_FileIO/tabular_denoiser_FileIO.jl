function GenerativeModelProtocols.load_tabular_denoiser_parameters(saved_model::String;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name,
    tabular_denoiser_group_name::String = GenerativeModelProtocols.@default_tabular_denoiser_group_name)

    return GenerativeModelProtocols.load_tabular_denoiser_parameters(FileIO.query(saved_model);
        main_group_name = main_group_name,
        generative_model_group_name = generative_model_group_name,
        tabular_denoiser_group_name = tabular_denoiser_group_name
    )
end

