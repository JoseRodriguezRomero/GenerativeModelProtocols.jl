function GenerativeModelProtocols.load_normalizing_flow_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String};
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    HDF5.h5open(file.filename, "r") do file
        velocity_field_group = file[main_group_name][generative_model_group_name]["velocity_field"]
        velocity_field, velocity_field_ps = read_group_chain_parameters(velocity_field_group)

        nf_model = GenerativeModelProtocols.NormalizingFlow(; velocity_field = velocity_field)
        nf_model._ps[] = merge(nf_model._ps[], (velocity_field = velocity_field_ps,))

        return nf_model
    end
end

function GenerativeModelProtocols._save_model(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, 
    model::GenerativeModelProtocols.NormalizingFlow;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    HDF5.h5open(file.filename, "w") do file
        main_group = HDF5.create_group(file, main_group_name)
        generative_model_group = HDF5.create_group(main_group, generative_model_group_name)
        velocity_field_group = HDF5.create_group(generative_model_group, "velocity_field")

        write_chain_to_file(model.velocity_field, model._ps[].velocity_field, velocity_field_group)
    end
end

