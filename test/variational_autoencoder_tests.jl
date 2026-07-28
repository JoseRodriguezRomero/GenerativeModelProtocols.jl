function test_compare_models(model_a::GenerativeModelProtocols.VariationalAutoencoder, model_b::GenerativeModelProtocols.VariationalAutoencoder)
    @test compare_chains(model_a.encoders, model_b.encoders)
    @test compare_chains(model_a.decoders, model_b.decoders)
end

@testset "VariationalAutoencoder Tests" begin
    @testset "Constructors Tests" begin
        input_size = 3
        latent_dim = 3

        model_1 = GenerativeModelProtocols.VariationalAutoencoder(input_size, latent_dim; β = 0.1)
        randomize_chains!(model_1.encoders)
        randomize_chains!(model_1.decoders)

        model_2 = GenerativeModelProtocols.VariationalAutoencoder(model_1.encoders, model_1.decoders; β = 0.2)
        model_3 = GenerativeModelProtocols.VariationalAutoencoder(;
            latent_dim      = latent_dim,
            latent_layers   = length(model_1.encoders),
            encoders        = model_1.encoders,
            decoders        = model_1.decoders,
            β               = [0.1, 0.2]
        )

        @test isa(model_1, GenerativeModelProtocols.VariationalAutoencoder)
        @test isa(model_2, GenerativeModelProtocols.VariationalAutoencoder)
        @test isa(model_3, GenerativeModelProtocols.VariationalAutoencoder)

        test_compare_models(model_1, model_2)
        test_compare_models(model_1, model_3)
        test_compare_models(model_2, model_3)
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(500)

        input_size = size(train_data,1)
        latent_dim = 2
        latent_layers = 2

        model = GenerativeModelProtocols.VariationalAutoencoder(input_size, latent_dim, latent_layers; β = [0.1, 0.2])
        test_train_model(model, train_data)
        test_train_model_no_data(model)
    end

    @testset "Incompatible Encoder/Decoder Architecture Test" begin
        hidden_layer_size = 32
        activation_function = relu

        function test_encoder_decoder(encoders, decoders)
            try
                GenerativeModelProtocols.VariationalAutoencoder(encoders, decoders)
                return false
            catch
                return true
            end
        end

        # Test incompatible input dimensions
        num_inputs_enc = 2
        num_inputs_dec = 3
        latent_dim = 1 

        encoder = Chain(
            Dense(num_inputs_enc => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => 2*latent_dim)
        ) |> f64
        decoder = Chain(
            Dense(latent_dim => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => num_inputs_dec)
        ) |> f64

        @test test_encoder_decoder((encoder,), (decoder,))

        # Test incompatible latent dimensions
        num_inputs = 2
        latent_dim_enc = 1
        latent_dim_dec = 2

        encoder = Chain(
            Dense(num_inputs => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => 2*latent_dim_enc)
        ) |> f64
        decoder = Chain(
            Dense(latent_dim_dec => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => num_inputs)
        ) |> f64

        @test test_encoder_decoder((encoder,), (decoder,))

        # Test incompatible latent layers
        num_inputs = 2
        latent_dim = 1 

        encoders = (
            Chain(
                Dense(num_inputs => hidden_layer_size, activation_function),
                Dense(hidden_layer_size => 2*latent_dim)
            ) |> f64,
            Chain(
                Dense(latent_dim => hidden_layer_size, activation_function),
                Dense(hidden_layer_size => 2*latent_dim)
            ),
            Chain(
                Dense(latent_dim => hidden_layer_size, activation_function),
                Dense(hidden_layer_size => 2*latent_dim)
            ),
            Chain(
                Dense(latent_dim => hidden_layer_size, activation_function),
                Dense(hidden_layer_size => 2*latent_dim)
            )
        )
        decoders = (
            Chain(
                Dense(latent_dim => hidden_layer_size, activation_function),
                Dense(hidden_layer_size => num_inputs)
            ) |> f64,
            Chain(
                Dense(latent_dim => hidden_layer_size, activation_function),
                Dense(hidden_layer_size => 2*latent_dim)
            ),
            Chain(
                Dense(latent_dim => hidden_layer_size, activation_function),
                Dense(hidden_layer_size => 2*latent_dim)
            )
        )

        @test test_encoder_decoder(encoders, decoders)

        # Test incompatible activation function encoder
        num_inputs = 2
        latent_dim = 1 

        encoder = Chain(
            Dense(num_inputs => hidden_layer_size, sin),
            Dense(hidden_layer_size => 2*latent_dim)
        ) |> f64

        decoder = Chain(
            Dense(latent_dim => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => num_inputs)
        ) |> f64

        @test test_encoder_decoder((encoder,), (decoder,))

        # Test incompatible activation function decoder
        num_inputs = 2
        latent_dim = 1 

        encoder = Chain(
            Dense(num_inputs => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => 2*latent_dim)
        ) |> f64

        decoder = Chain(
            Dense(latent_dim => hidden_layer_size, sin),
            Dense(hidden_layer_size => num_inputs)
        ) |> f64

        @test test_encoder_decoder((encoder,), (decoder,))
    end

    @testset "Make Synthetic Data Test" begin
        input_size = 3
        latent_dim = 3
        model = GenerativeModelProtocols.VariationalAutoencoder(input_size, latent_dim)
        test_model_make_synthetic_data(model)
    end

    @testset "Encode/Decode Test" begin
        input_size = 3
        latent_dim = 2
        latent_layers = 2
        model = GenerativeModelProtocols.VariationalAutoencoder(input_size, latent_dim, latent_layers)
        protocol = GenerativeModelProtocol(model)

        # Vector single call
        x = protocol()
        z = encode(model, x)
        x̂ = decode(model, z)

        @test isa(x, Vector)
        @test isa(z, Vector)

        @test isa(x̂, Vector)
        @test length(x) == length(x̂)

        # Matrix batch call
        x = protocol(100)
        z = encode(model, x)
        x̂ = decode(model, z)

        @test isa(x, Matrix)
        @test isa(z, Matrix)

        @test isa(x̂, Matrix)
        @test size(x) == size(x̂)
        @test size(z) == (latent_dim, size(x,2))
    end

    @testset "Display Test" begin
        input_size = 3
        latent_dim = 2
        
        model = GenerativeModelProtocols.VariationalAutoencoder(input_size, latent_dim)
        test_model_display(model)
        test_model_display(model, input_size)
    end

    @testset "Save and Load Test" begin
        input_size = 3
        latent_dim = 2
        latent_layers = 2
        model = GenerativeModelProtocols.VariationalAutoencoder(input_size, latent_dim, latent_layers)

        randomize_chains!(model.encoders)
        randomize_chains!(model.decoders)

        save_path = joinpath(@__DIR__(), "test_vae_model.h5")
        test_model_save(save_path, model)
        
        loaded_model = GenerativeModelProtocols.VariationalAutoencoder(save_path)
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

