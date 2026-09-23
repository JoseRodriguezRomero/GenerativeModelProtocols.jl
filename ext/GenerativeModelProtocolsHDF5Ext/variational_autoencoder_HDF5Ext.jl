function GenerativeModelProtocols.load_variational_autoencoder_parameters(file::FileIO.File{FileIO.DataFormat{:HDF5},String};
    main_group_name::String = GenerativeModelProtocols.@default_main_group_name,
    generative_model_group_name::String = GenerativeModelProtocols.@default_generative_model_group_name)

    HDF5.h5open(file.filename, "r") do file
        encoder_group = file[main_group_name][generative_model_group_name]["encoders"]
        decoder_group = file[main_group_name][generative_model_group_name]["decoders"]

        encoder_keys = keys(encoder_group)
        decoder_keys = keys(decoder_group)

        encoders = Vector{Chain}(undef, length(encoder_keys))
        decoders = Vector{Chain}(undef, length(decoder_keys))

        decoders_ps = Vector{NamedTuple}(undef, length(decoder_keys))
        encoders_ps = Vector{NamedTuple}(undef, length(encoder_keys))


        for i in eachindex(encoders)
            encoder_chain_group = encoder_group[encoder_keys[i]]
            encoder, encoder_ps = read_group_chain_parameters(encoder_chain_group)
            encoders[i] = encoder
            encoders_ps[i] = encoder_ps
        end

        for i in eachindex(decoders)
            decoder_chain_group = decoder_group[decoder_keys[i]]
            decoder, decoder_ps = read_group_chain_parameters(decoder_chain_group)
            decoders[i] = decoder
            decoders_ps[i] = decoder_ps
        end

        vae_model = GenerativeModelProtocols.VariationalAutoencoder(Tuple(encoders), Tuple(decoders))

        encoders_ps = NamedTuple{keys(vae_model._ps[].encoders)}(Tuple(encoders_ps))
        decoders_ps = NamedTuple{keys(vae_model._ps[].decoders)}(Tuple(decoders_ps))

        vae_model._ps[] = merge(vae_model._ps[], (
            encoders = encoders_ps,
            decoders = decoders_ps
        ))

        return vae_model
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

        function write_encoder_to_file(encoders::NamedTuple, encoders_ps::NamedTuple)
            for key in keys(encoders)
                encoder_group = HDF5.create_group(encoders_group, "$key")
                write_chain_to_file(encoders[key], encoders_ps[key], encoder_group)
            end
        end

        function write_decoder_to_file(decoders::NamedTuple, decoders_ps::NamedTuple)
            for key in keys(decoders)
                decoder_group = HDF5.create_group(decoders_group, "$key")
                write_chain_to_file(decoders[key], decoders_ps[key], decoder_group)
            end
        end

        write_encoder_to_file(model.encoders, model._ps[].encoders)
        write_decoder_to_file(model.decoders, model._ps[].decoders)
    end
end

