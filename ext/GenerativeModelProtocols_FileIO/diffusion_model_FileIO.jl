function GenerativeModelProtocols.load_diffusion_model_parameters(saved_model::String)
    return GenerativeModelProtocols.load_diffusion_model_parameters(FileIO.query(saved_model))
end

