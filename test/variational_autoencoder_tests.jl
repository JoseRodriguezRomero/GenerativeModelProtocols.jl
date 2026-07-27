@testset "VariationalAutoencoder Tests" begin
    @testset "Constructors Tests" begin
        input_dim = 3
        latent_dim = 3

        model_1 = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim; β = 0.1)
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
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(500)

        input_dim = size(train_data,1)
        latent_dim = 2

        model = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim)
        protocol = GenerativeModelProtocol(model, train_data;
            batchsize   = 32,
            epochs      = 20,
            optimiser   = Adam(; eta = 1.0E-3, beta = (0.95, 0.999)),
            device      = cpu_device()
        )

        train_log = train!(protocol)
        @test isa(train_log, typeof(protocol._log))
    end

    @testset "Make Synthetic Data Test" begin
        input_dim = 3
        latent_dim = 3
        model = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim)
        protocol = GenerativeModelProtocol(model)

        x_synthetic = protocol(100)
        @test isa(x_synthetic, Matrix)
    end

    @testset "Encode/Decode Test" begin
        input_dim = 3
        latent_dim = 2
        latent_layers = 2
        model = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim, latent_layers)

        x = rand(Float64, input_dim, 100)
        z = encode(model, x)
        x̂ = decode(model, z)

        @test isa(x, Matrix)
        @test isa(z, Matrix)

        @test isa(x̂, Matrix)
        @test size(x) == size(x̂)
        @test size(z) == (latent_dim, size(x,2))
    end

    @testset "Save and Load Test" begin
        input_dim = 3
        latent_dim = 2
        latent_layers = 2
        model = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim, latent_layers)
        protocol = GenerativeModelProtocol(model)

        randomize_chains!(model.encoders)
        randomize_chains!(model.decoders)

        save_path = joinpath(@__DIR__(), "test_vae_model.h5")
        save(save_path, protocol)
        @test check_file_size(save_path)
        
        loaded_model = GenerativeModelProtocols.VariationalAutoencoder(save_path)
        @test compare_chains(model.encoders, loaded_model.encoders)
        @test compare_chains(model.decoders, loaded_model.decoders)

        remove_file(save_path)
    end
end

