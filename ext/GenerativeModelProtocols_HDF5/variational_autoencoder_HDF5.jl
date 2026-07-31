function GenerativeModelProtocols.load_variational_autoencoder_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String};
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    HDF5.h5open(file.filename, "r") do file
        encoder_group = file[main_group_name][generative_model_group_name]["encoders"]
        decoder_group = file[main_group_name][generative_model_group_name]["decoders"]

        encoders = Vector{Chain}(undef, length(keys(encoder_group)))
        decoders = Vector{Chain}(undef, length(keys(decoder_group)))

        for i in eachindex(encoders)
            encoder_chain_group = encoder_group["encoder $i"]
            encoders[i] = read_group_chain_parameters(encoder_chain_group)
        end

        for i in eachindex(decoders)
            decoder_chain_group = decoder_group["decoder $i"]
            decoders[i] = read_group_chain_parameters(decoder_chain_group)
        end

        return GenerativeModelProtocols.VariationalAutoencoder(Tuple(encoders), Tuple(decoders))
    end
end

function GenerativeModelProtocols._save_model(file::FileIO.File{FileIO.DataFormat{:HDF5},String}, 
    model::GenerativeModelProtocols.VariationalAutoencoder;
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)
    
    HDF5.h5open(file.filename, "w") do file
        main_group = HDF5.create_group(file, main_group_name)
        generative_model_group = HDF5.create_group(main_group, generative_model_group_name)
        encoders_group = HDF5.create_group(generative_model_group, "encoders")
        decoders_group = HDF5.create_group(generative_model_group, "decoders")        

        function write_encoder_to_file(encoder::Tuple)
            for i in eachindex(encoder)
                encoder_group = HDF5.create_group(encoders_group, "encoder $i")
                write_chain_to_file(encoder[i], encoder_group)
            end
        end

        function write_decoder_to_file(decoder::Tuple)
            for i in eachindex(decoder)
                decoder_group = HDF5.create_group(decoders_group, "decoder $i")
                write_chain_to_file(decoder[i], decoder_group)
            end
        end

        write_encoder_to_file(model.encoders)
        write_decoder_to_file(model.decoders)
    end
end

