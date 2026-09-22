include("utility_utils/tests_base_utils.jl")

@testset "GenerativeAdversarialNetwork Tests" begin
    @testset "Constructors Tests" begin
        input_size = 3
        latent_dim = 3

        model_1 = GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size, latent_dim)
        randomize_generative_adversarial_network!(model_1)

        model_2 = GenerativeModelProtocols.GenerativeAdversarialNetwork(;
            discriminator = model_1.discriminator,
            vae_model     = model_1.vae_model
        )

        model_2._ps_discriminator[] = merge(model_2._ps_discriminator[], model_1._ps_discriminator[])
        model_2.vae_model._ps[] = merge(model_2.vae_model._ps[], model_1.vae_model._ps[])

        @test isa(model_1, GenerativeModelProtocols.GenerativeAdversarialNetwork)
        @test isa(model_2, GenerativeModelProtocols.GenerativeAdversarialNetwork)

        test_compare_models(model_1, model_2)
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(50)

        input_size = size(train_data,1)
        latent_dim = 2

        model = GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size, latent_dim)
        test_train_model(model, train_data; grad_penalty = true, weight_clipping = true)
        test_train_model_no_data(model, input_size; grad_penalty = true, weight_clipping = true)
        test_train_model_reactant(model, train_data; grad_penalty = true, weight_clipping = true)
    end

    @testset "Incompatible GenerativeAdversarialNetwork Architecture Test" begin
        # Test incompatible VAE input size
        discriminator = example_critic(1,1) 
        vae_model = example_vae(2,2)
        @test_throws Exception GenerativeModelProtocols.GenerativeAdversarialNetwork(;
            discriminator = discriminator, 
            vae_model     = vae_model
        )

        # Test incompatible discriminator output size
        discriminator = example_critic(2,2) 
        vae_model = example_vae(2,2)
        @test_throws Exception GenerativeModelProtocols.GenerativeAdversarialNetwork(;
            discriminator = discriminator, 
            vae_model     = vae_model
        )
    end

    @testset "Make Synthetic Data Test" begin
        input_size = 3
        latent_dim = 3
        model = GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size, latent_dim)
        test_model_make_synthetic_data(model, input_size)
    end

    @testset "Encode/Decode Test" begin
        input_size = 3
        latent_dim = 2
        model = GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size, latent_dim)
        protocol = make_empty_data_prot(model, input_size)

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
        
        model = GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size, latent_dim)
        test_model_display(model, input_size)
        test_model_display_no_data(model, input_size)
    end

    @testset "Save and Load Test" begin
        input_size = 3
        latent_dim = 2
        latent_layers = 2
        model = GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size, latent_dim, latent_layers)
        randomize_generative_adversarial_network!(model)

        save_path = joinpath(@__DIR__(), "test_gan_model.h5")
        test_model_save(save_path, model, input_size)
        
        loaded_protocol = GenerativeModelProtocol(save_path)
        loaded_model = loaded_protocol.model
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

