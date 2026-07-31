function test_compare_tabular_denoiser(denoiser_a::GenerativeModelProtocols.TabularDenoiser, denoiser_b::GenerativeModelProtocols.TabularDenoiser)
    ϵ = 1.0E-9

    @test denoiser_a.T == denoiser_b.T
    @test compare_chains(denoiser_a.time_embedding_mlp,denoiser_b.time_embedding_mlp)
    @test compare_layers(denoiser_a.input_projection, denoiser_b.input_projection)
    @test compare_layers(denoiser_a.residual_layers, denoiser_b.residual_layers)
    @test compare_layers(denoiser_a.time_projection_layers, denoiser_b.time_projection_layers)
    @test compare_layers(denoiser_a.output_projection, denoiser_b.output_projection)
    @test abs(denoiser_a.max_period - denoiser_b.max_period) < ϵ
end

function test_compare_models(model_a::GenerativeModelProtocols.DiffusionModel, model_b::GenerativeModelProtocols.DiffusionModel)
    ϵ = 1.0E-9

    @test model_a.T == model_b.T
    @test maximum(model_a.α - model_b.α) < ϵ
    @test maximum(model_a.ᾱ - model_b.ᾱ) < ϵ
    @test maximum(model_a.β - model_b.β) < ϵ
    test_compare_tabular_denoiser(model_a.denoiser_model, model_b.denoiser_model)
end

function randomize_tabular_denoiser!(denoiser::GenerativeModelProtocols.TabularDenoiser)
    randomize_chains!(denoiser.time_embedding_mlp)
    randomize_layers!(denoiser.input_projection)
    randomize_layers!(denoiser.residual_layers)
    randomize_layers!(denoiser.time_projection_layers)
    randomize_layers!(denoiser.output_projection)
end

function randomize_diffusion_model!(model::GenerativeModelProtocols.DiffusionModel)
    randomize_tabular_denoiser!(model.denoiser_model)
end

@testset "DiffusionModel Tests" begin
    @testset "Constructors Tests" begin
        input_size = 3
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        @test isa(model, GenerativeModelProtocols.DiffusionModel)
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(500)

        input_size = size(train_data,1)
        T = 21

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        test_train_model(model, train_data)
        test_train_model_no_data(model)
    end

    @testset "Make Synthetic Data Test" begin
        input_size = 2
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        test_model_make_synthetic_data(model)
    end

    @testset "Display Test" begin
        input_size = 3
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        test_model_display(model)
        test_model_display(model, input_size)
        test_display(model.denoiser_model)
    end

    @testset "Save and Load Test" begin
        input_size = 3
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        randomize_diffusion_model!(model)

        save_path = joinpath(@__DIR__(), "test_dm_model.h5")
        test_model_save(save_path, model)
        
        loaded_model = GenerativeModelProtocols.DiffusionModel(save_path)
        test_compare_models(model, loaded_model)

        model_β = loaded_model.β
        model_denoiser = GenerativeModelProtocols.TabularDenoiser(save_path)
        loaded_model = GenerativeModelProtocols.DiffusionModel(model_β, model_denoiser)
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

