function GenerativeModelProtocols.load_gaussian_mixture_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String}; 
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    HDF5.h5open(file.filename, "r") do file
        gaussian_mixture_model_parameters_group = file[main_group_name][generative_model_group_name]
        predictor_network_group = gaussian_mixture_model_parameters_group["predictor_network"]
        predictor_network = GenerativeModelProtocols.ChainParameters(read_group_layer_parameters(predictor_network_group))
        k = HDF5.attrs(gaussian_mixture_model_parameters_group)["k"]
        log_σ² = read(gaussian_mixture_model_parameters_group["log_sigma_squared"])
        μ = read(gaussian_mixture_model_parameters_group["mu"])
        p = read(gaussian_mixture_model_parameters_group["p"])

        return GenerativeModelProtocols.GaussianMixtureModelParameters(k, log_σ², μ, predictor_network, p)
    end
end

function GenerativeModelProtocols._save(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, 
    model::GenerativeModelProtocols.GaussianMixtureModel;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    gaussian_mixture_model_parameters = GenerativeModelProtocols.gaussian_mixture_model_parameters(model)
    HDF5.h5open(file.filename, "w") do file
        main_group = HDF5.create_group(file, main_group_name)
        generative_model_group = HDF5.create_group(main_group, generative_model_group_name)
        predictor_network_group = HDF5.create_group(generative_model_group, "predictor_network")

        HDF5.write_attribute(generative_model_group, "k", gaussian_mixture_model_parameters.k)
        HDF5.write(generative_model_group, "log_sigma_squared", gaussian_mixture_model_parameters.log_σ²)
        HDF5.write(generative_model_group, "mu", gaussian_mixture_model_parameters.μ)
        write_chain_to_file(gaussian_mixture_model_parameters.predictor_network, predictor_network_group)
        HDF5.write(generative_model_group, "p", gaussian_mixture_model_parameters.p)
    end
end

