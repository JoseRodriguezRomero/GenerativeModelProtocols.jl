function GenerativeModelProtocols.load_generative_adversarial_network_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String}; 
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    discriminator, discriminator_ps = HDF5.h5open(file.filename, "r") do file
        discriminator_group = file[main_group_name][generative_model_group_name]["discriminator"]
        return read_group_chain_parameters(discriminator_group)
    end

    vae_model = GenerativeModelProtocols.load_variational_autoencoder_parameters(file;
        main_group_name             = main_group_name,
        generative_model_group_name = generative_model_group_name * "/vae_model"
    )

    gan_model = GenerativeModelProtocols.GenerativeAdversarialNetwork(;
        discriminator = discriminator, 
        vae_model     = vae_model
    )
    gan_model._ps_discriminator[] = merge(gan_model._ps_discriminator[],(
        discriminator = discriminator_ps,
    ))

    return gan_model
end

function GenerativeModelProtocols._save_model(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, 
    model::GenerativeModelProtocols.GenerativeAdversarialNetwork;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    GenerativeModelProtocols._save_model(file, model.vae_model;
        main_group_name             = main_group_name,
        generative_model_group_name = generative_model_group_name * "/vae_model"
    )

    HDF5.h5open(file.filename, "r+") do file
        main_group = file[main_group_name]
        generative_model_group = main_group[generative_model_group_name]
        discriminator_group = HDF5.create_group(generative_model_group, "discriminator")

        write_chain_to_file(model.discriminator, model._ps_discriminator[].discriminator, discriminator_group)
    end
end

