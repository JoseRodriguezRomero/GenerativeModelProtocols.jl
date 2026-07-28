module GenerativeModelProtocols_FileIO

import GenerativeModelProtocols
import FileIO

function GenerativeModelProtocols._save_metadata(file_name::String, protocol::GenerativeModelProtocols.GenerativeModelProtocol; 
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name, 
    metadata_group_name::String = GenerativeModelProtocols.@default_metadata_group_name, 
    metadata::Union{Dict{String,Any}, NamedTuple, Nothing} = nothing)

    if isnothing(metadata)
        metadata = Dict{String,Any}()
    elseif isa(metadata, NamedTuple)
        metadata = Dict(string(k) => v for (k, v) in pairs(metadata))
    end

    GenerativeModelProtocols._save_metadata(FileIO.query(file_name), protocol; 
        main_group_name = main_group_name, 
        metadata_group_name = metadata_group_name, 
        metadata = metadata
    )
end

function GenerativeModelProtocols._save_model(file_name::String, model::GenerativeModelProtocols.AbstractGenerativeModel;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    GenerativeModelProtocols._save_model(FileIO.query(file_name), model; 
        main_group_name = main_group_name, 
        generative_model_group_name = generative_model_group_name
    )
end

"""
    FileIO.save(file_name::String, protocol::GenerativeModelProtocols.GenerativeModelProtocol; main_group_name::String, metadata_group_name::String, generative_model_group_name::String, metadata::Union{Dict{String,Any}, NamedTuple, Nothing})
    
Convenience function for saving already trained generative models. End users can
use this convenience function the exact same way regardless of the generative 
model architecture.

# Arguments
* `main_group_name::String`:
* `metadata_group_name::String`:
* `generative_model_group_name::String`:
* `metadata::Union{Dict{String,Any}, NamedTuple, Nothing}`:

"""
function FileIO.save(file_name::String, protocol::GenerativeModelProtocols.GenerativeModelProtocol; 
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name, 
    metadata_group_name::String = GenerativeModelProtocols.@default_metadata_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name,
    metadata::Union{Dict{String,Any}, NamedTuple, Nothing} = nothing)

    GenerativeModelProtocols._save(file_name, protocol; 
        main_group_name = main_group_name, 
        metadata_group_name = metadata_group_name, 
        generative_model_group_name = generative_model_group_name,
        metadata = metadata
    )
end

include("diffusion_model_FileIO.jl")
include("gaussian_mixture_model_FileIO.jl")
include("variational_autoencoder_FileIO.jl")

end

