function GenerativeModelProtocols.load_variational_autoencoder_parameters(saved_model::String)
    return GenerativeModelProtocols.load_variational_autoencoder_parameters(FileIO.query(saved_model))
end

