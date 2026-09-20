@testset "GaussianMixtureModel Tests" begin
    function test_compare_models(model_a::GenerativeModelProtocols.GaussianMixtureModel, model_b::GenerativeModelProtocols.GaussianMixtureModel)
        ϵ = 1.0E-8
        
        @test maximum(abs.(model_a.log_σ² - model_b.log_σ²)) < ϵ
        @test maximum(abs.(model_a.μ - model_b.μ)) < ϵ
        @test compare_chains(model_a._ps[].predictor_network, model_b._ps[].predictor_network)
        @test maximum(abs.(model_a.p - model_b.p)) < ϵ
    end

    function example_chain(input_size::Int, output_size::Int; hidden_layer_size::Int = 16, activation_function::Function = relu)
        return Chain(
            Dense(input_size => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => output_size),
        )
    end

    function example_gmm(k::Int, input_size::Int; 
        log_σ²::Union{Matrix{Float32}, Nothing} = nothing, 
        μ::Union{Matrix{Float32}, Nothing} = nothing, 
        predictor_network::Union{Chain, Nothing} = nothing,
        p::Union{Vector{Float32}, Nothing} = nothing)

        if isnothing(log_σ²)
            log_σ² = rand(Float32, k, input_size)
        end

        if isnothing(μ)
            μ = rand(Float32, k, input_size)
        end

        if isnothing(predictor_network)
            predictor_network = example_chain(input_size, k)
        end

        if isnothing(p)
            p = rand(Float32, k)
        end

        return GenerativeModelProtocols.GaussianMixtureModel(;
            log_σ²              = log_σ²,
            μ                   = μ,
            predictor_network   = predictor_network,
            p                   = p
        )
    end

    @testset "Constructors Tests" begin
        k = 15
        input_size = 3

        model_1 = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        model_2 = GenerativeModelProtocols.GaussianMixtureModel(;
            log_σ²              = model_1.log_σ²,
            μ                   = model_1.μ,
            predictor_network   = model_1.predictor_network,
            p                   = model_1.p,
            _ps                 = model_1._ps,
            _st                 = model_1._st
        )

        @test isa(model_1, GenerativeModelProtocols.GaussianMixtureModel)
        @test isa(model_2, GenerativeModelProtocols.GaussianMixtureModel)

        test_compare_models(model_1, model_2)
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(50)

        k = 15
        input_size = size(train_data,1)

        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        test_train_model(model, train_data)
        test_train_model_no_data(model, input_size)
        test_train_model_reactant(model, train_data)
    end

    @testset "Incompatible GaussianMixtureModel Architecture Test" begin
        # Test incompatible input size
        @test_throws Exception example_gmm(5, 2; log_σ² = rand(Float32, 5, 3))
        @test_throws Exception example_gmm(5, 2; μ = rand(Float32, 5, 3))
        @test_throws Exception example_gmm(5, 2; predictor_network = example_chain(3, 5))

        # Test incompatible k
        @test_throws Exception example_gmm(5, 2; log_σ² = rand(Float32, 4, 2))
        @test_throws Exception example_gmm(5, 2; μ = rand(Float32, 4, 2))
        @test_throws Exception example_gmm(5, 2; p = rand(Float32, 4))
        @test_throws Exception example_gmm(5, 2; predictor_network = example_chain(2, 4))        
    end

    @testset "Make Synthetic Data Test" begin
        k = 15
        input_dims = 3

        model = GenerativeModelProtocols.GaussianMixtureModel(input_dims, k)
        test_model_make_synthetic_data(model, input_dims)
        test_model_make_categorical_synthetic_data(model, input_dims)

        # Test input and latent size
        protocol = make_empty_data_prot(model, input_dims)
        @test input_size(protocol) != latent_size(protocol)
    end

    @testset "Categorize Test" begin
        k = 15
        input_size = 3

        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        test_model_categorize(model, input_size)
    end

    @testset "Display Test" begin
        k = 15
        input_size = 3

        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        test_model_display_no_data(model, input_size)
        test_model_display(model, input_size)
    end

    @testset "Save and Load Test" begin
        k = 15
        input_size = 3
        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)

        randomize_chain!(model._ps[].predictor_network)

        save_path = joinpath(@__DIR__(), "test_gmm_model.h5")
        test_model_save(save_path, model, input_size)
        
        loaded_protocol = GenerativeModelProtocol(save_path)
        loaded_model = loaded_protocol.model
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

