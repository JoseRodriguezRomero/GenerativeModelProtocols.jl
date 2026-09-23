function GenerativeModelProtocols.load_normalizing_flow_parameters(saved_model::String;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    return GenerativeModelProtocols.load_normalizing_flow_parameters(FileIO.query(saved_model);
        main_group_name             = main_group_name,
        generative_model_group_name = generative_model_group_name
    )
end

