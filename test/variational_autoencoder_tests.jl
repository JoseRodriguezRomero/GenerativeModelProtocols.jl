function test_compare_models(model_a::GenerativeModelProtocols.VariationalAutoencoder, model_b::GenerativeModelProtocols.VariationalAutoencoder)
    @test compare_chains(model_a.encoders, model_b.encoders)
    @test compare_chains(model_a.decoders, model_b.decoders)
end

@testset "VariationalAutoencoder Tests" begin
    @testset "Constructors Tests" begin
        input_dim = 3
        latent_dim = 3

        model_1 = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim; β = 0.1)
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

        input_dim = size(train_data,1)
        latent_dim = 2

        model = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim)
        test_train_model(model, train_data)
    end

    @testset "Make Synthetic Data Test" begin
        input_dim = 3
        latent_dim = 3
        model = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim)
        test_model_make_synthetic_data(model)
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

    @testset "Display Test" begin
        input_dim = 3
        latent_dim = 2
        
        model = GenerativeModelProtocols.VariationalAutoencoder(input_dim, latent_dim)
        test_model_display(model)
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
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

