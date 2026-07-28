function test_compare_models(model_a::GenerativeModelProtocols.GaussianMixtureModel, model_b::GenerativeModelProtocols.GaussianMixtureModel)
    ϵ = 1.0E-9
    
    @test model_a.k == model_b.k
    @test maximum(abs.(model_a.log_σ² - model_b.log_σ²)) < ϵ
    @test maximum(abs.(model_a.μ - model_b.μ)) < ϵ
    @test compare_chains(model_a.predictor_network, model_b.predictor_network)
    @test maximum(abs.(model_a.p - model_b.p)) < ϵ
end

@testset "GaussianMixtureModel Tests" begin
    @testset "Constructors Tests" begin
        k = 15
        input_size = 3

        model_1 = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        model_2 = GenerativeModelProtocols.GaussianMixtureModel(;
            k                   = model_1.k,
            log_σ²              = model_1.log_σ²,
            μ                   = model_1.μ,
            predictor_network   = model_1.predictor_network,
            p                   = model_1.p
        )

        @test isa(model_1, GenerativeModelProtocols.GaussianMixtureModel)
        @test isa(model_2, GenerativeModelProtocols.GaussianMixtureModel)

        test_compare_models(model_1, model_2)
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(500)

        k = 15
        input_size = size(train_data,1)

        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        test_train_model(model, train_data)
        test_train_model_no_data(model)
    end

    @testset "Make Synthetic Data Test" begin
        k = 15
        input_size = 3

        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        test_model_make_synthetic_data(model)
        test_model_make_categorical_synthetic_data(model)
    end

    @testset "Categorize Test" begin
        k = 15
        input_size = 3

        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        test_model_categorize(model)
    end

    @testset "Display Test" begin
        k = 15
        input_size = 3

        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)
        test_model_display(model)
        test_model_display(model, input_size)
    end

    @testset "Save and Load Test" begin
        k = 15
        input_size = 3
        model = GenerativeModelProtocols.GaussianMixtureModel(input_size, k)

        randomize_chains!(model.predictor_network)

        save_path = joinpath(@__DIR__(), "test_gmm_model.h5")
        test_model_save(save_path, model)
        
        loaded_model = GenerativeModelProtocols.GaussianMixtureModel(save_path)
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

