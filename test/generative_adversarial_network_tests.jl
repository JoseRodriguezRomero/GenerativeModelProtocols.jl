@testset "GenerativeAdversarialNetwork Tests" begin
    function test_compare_models(model_a::GenerativeModelProtocols.GenerativeAdversarialNetwork, model_b::GenerativeModelProtocols.GenerativeAdversarialNetwork)
        @test compare_chains(model_a.discriminator, model_b.discriminator)
        @test compare_chains(model_a.vae_model.encoders, model_b.vae_model.encoders)
        @test compare_chains(model_a.vae_model.decoders, model_b.vae_model.decoders)
    end

    function randomize_generative_adversarial_network!(model::GenerativeModelProtocols.GenerativeAdversarialNetwork)
        randomize_chains!(model.discriminator)
        randomize_chains!(model.vae_model.encoders)
        randomize_chains!(model.vae_model.decoders)
    end

    function example_encoder(num_inputs::Int, latent_dim::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
        return Chain(
            Dense(num_inputs => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => 2*latent_dim, activation_function)
        )
    end

    function example_generator(num_inputs::Int, latent_dim::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
        return Chain(
            Dense(latent_dim => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => num_inputs, activation_function)
        )
    end

    function example_vae(num_inputs::Int, latent_dim::Int; hidden_layer_size::Int = 32, activation_function::Function = relu)
        encoder = example_encoder(num_inputs, latent_dim;
            hidden_layer_size   = hidden_layer_size,
            activation_function = activation_function
        )

        decoder = example_generator(num_inputs, latent_dim;
            hidden_layer_size   = hidden_layer_size,
            activation_function = activation_function
        )

        return GenerativeModelProtocols.VariationalAutoencoder((encoder,), (decoder,))
    end

    function example_critic(num_inputs::Int, output_size::Int = 1; hidden_layer_size::Int = 32, activation_function::Function = relu)
        return Chain(
            Dense(num_inputs => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => output_size, activation_function)
        )
    end

    @testset "Constructors Tests" begin
        input_size = 3
        latent_dim = 3

        model_1 = GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size, latent_dim)
        randomize_generative_adversarial_network!(model_1)

        model_2 = GenerativeModelProtocols.GenerativeAdversarialNetwork(;
            discriminator = model_1.discriminator,
            vae_model     = model_1.vae_model
        )

        @test isa(model_1, GenerativeModelProtocols.GenerativeAdversarialNetwork)
        @test isa(model_2, GenerativeModelProtocols.GenerativeAdversarialNetwork)

        test_compare_models(model_1, model_2)
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(500)

        input_size = size(train_data,1)
        latent_dim = 2

        model = GenerativeModelProtocols.GenerativeAdversarialNetwork(input_size, latent_dim)
        test_train_model(model, train_data; grad_penalty = true, weight_clipping = true)
        test_train_model_no_data(model, input_size; grad_penalty = true, weight_clipping = true)
    end

    @testset "Incompatible GenerativeAdversarialNetwork Architecture Test" begin
        GAN = GenerativeModelProtocols.GenerativeAdversarialNetwork

        # Test incompatible VAE input size
        discriminator = example_critic(1,1) 
        vae_model = example_vae(2,2)
        @test_throws Exception GAN(discriminator, vae_model)

        # Test incompatible discriminator output size
        discriminator = example_critic(2,2) 
        vae_model = example_vae(2,2)
        @test_throws Exception GAN(discriminator, vae_model)
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
    end
end

