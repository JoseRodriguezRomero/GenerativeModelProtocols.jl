@testset "VariationalAutoencoder Tests" begin
    function test_compare_models(model_a::GenerativeModelProtocols.VariationalAutoencoder, model_b::GenerativeModelProtocols.VariationalAutoencoder)
        @test compare_chains(model_a.encoders, model_b.encoders)
        @test compare_chains(model_a.decoders, model_b.decoders)
    end

    function randomize_variational_autoencoder!(model::GenerativeModelProtocols.VariationalAutoencoder)
        randomize_chains!(model.encoders)
        randomize_chains!(model.decoders)
    end

    function example_encoder(num_inputs::Int, latent_dim::Int, latent_layers::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
        first_layer = Chain(
            Dense(num_inputs => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => 2*latent_dim)
        )

        lower_layer = Chain(
            Dense(latent_dim => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => 2*latent_dim)
        )

        lower_layers = [lower_layer for _ in 1:(latent_layers-1)]
        return Tuple(vcat([first_layer], lower_layers))
    end

    function example_decoder(num_inputs::Int, latent_dim::Int, latent_layers::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
        first_layer = Chain(
            Dense(latent_dim => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => num_inputs)
        )

        lower_layer = Chain(
            Dense(latent_dim => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => 2*latent_dim)
        )

        lower_layers = [lower_layer for _ in 1:(latent_layers-1)]
        return Tuple(vcat([first_layer], lower_layers))
    end

    @testset "Constructors Tests" begin
        input_size = 3
        latent_dim = 3

        model_1 = GenerativeModelProtocols.VariationalAutoencoder(input_size, latent_dim)
        randomize_variational_autoencoder!(model_1)

        model_2 = GenerativeModelProtocols.VariationalAutoencoder(model_1.encoders, model_1.decoders)
        model_3 = GenerativeModelProtocols.VariationalAutoencoder(;
            latent_dim      = latent_dim,
            latent_layers   = length(model_1.encoders),
            encoders        = model_1.encoders,
            decoders        = model_1.decoders
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

        model = GenerativeModelProtocols.VariationalAutoencoder(input_size, latent_dim, latent_layers)
        test_train_model(model, train_data; β = [0.1, 0.2])
        test_train_model_no_data(model; β = [0.1, 0.2])
    end

    @testset "Incompatible VariationalAutoencoder Architecture Test" begin
        VAE = GenerativeModelProtocols.VariationalAutoencoder

        # Test incompatible input dimensions
        encoders = example_encoder(2, 2, 3)
        decoders = example_decoder(1, 2, 3)
        @test_throws Exception VAE(encoders, decoders)

        # Test incompatible latent dimensions
        encoders = example_encoder(2, 2, 3)
        decoders = example_decoder(2, 1, 3)
        @test_throws Exception VAE(encoders, decoders)

        # Test incompatible latent layers
        encoders = example_encoder(2, 2, 3)
        decoders = example_decoder(2, 2, 2)
        @test_throws Exception VAE(encoders, decoders)

        # Test incompatible activation function encoder
        encoders = example_encoder(2, 2, 3; activation_function = sin)
        decoders = example_decoder(2, 2, 3)
        @test_throws Exception VAE(encoders, decoders)

        # Test incompatible activation function decoder
        encoders = example_encoder(2, 2, 3)
        decoders = example_decoder(2, 2, 3; activation_function = sin)
        @test_throws Exception VAE(encoders, decoders)
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
        z = encode(protocol, x)
        x̂ = decode(protocol, z)

        @test isa(x, Vector)
        @test isa(z, Vector)

        @test isa(x̂, Vector)
        @test length(x) == length(x̂)

        # Matrix batch call
        x = protocol(100)
        z = encode(protocol, x)
        x̂ = decode(protocol, z)

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
        randomize_variational_autoencoder!(model)

        save_path = joinpath(@__DIR__(), "test_vae_model.h5")
        test_model_save(save_path, model)
        
        loaded_model = GenerativeModelProtocols.VariationalAutoencoder(save_path)
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

