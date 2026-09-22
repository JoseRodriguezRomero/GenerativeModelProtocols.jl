include("utility_utils/tests_base_utils.jl")

@testset "DiffusionModel Tests" begin
    @testset "Constructors Tests" begin
        input_size = 3
        T = 30

        model1 = GenerativeModelProtocols.DiffusionModel(input_size, T)
        model2 = GenerativeModelProtocols.DiffusionModel(input_size, model1.β)

        @test isa(model1, GenerativeModelProtocols.DiffusionModel)
        @test isa(model2, GenerativeModelProtocols.DiffusionModel)
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(50)

        input_size = size(train_data,1)
        T = 21

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        test_train_model(model, train_data)
        test_train_model_no_data(model, input_size)
        # test_train_model_reactant(model, train_data)
    end

    @testset "Incompatible DiffusionModel Architecture Test" begin
        num_inputs = 2
        default_T1 = 30
        default_dm1 = GenerativeModelProtocols.DiffusionModel(num_inputs, default_T1)

        default_T2 = default_T1 + 1
        default_dm2 = GenerativeModelProtocols.DiffusionModel(num_inputs, default_T2)
        
        # Test incompatible vector lengths
        @test_throws Exception GenerativeModelProtocols.DiffusionModel(;
            β              = default_dm1.β,
            denoiser_model = default_dm2.denoiser_model
        )

        @test_throws Exception GenerativeModelProtocols.DiffusionModel()
    end

    @testset "Make Synthetic Data Test" begin
        input_size = 2
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        test_model_make_synthetic_data(model, input_size)
    end

    @testset "Encode/Decode Test" begin
        input_dims = 3
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_dims, T)
        protocol = make_empty_data_prot(model, input_dims)

        # Test input and latent sizes
        @test input_size(protocol) == latent_size(protocol)

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
        @test size(z) == (input_dims, size(x,2))
    end

    @testset "Display Test" begin
        input_size = 3
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        test_model_display_no_data(model, input_size)
        test_model_display(model, input_size)
        test_display(model.denoiser_model)
    end

    @testset "Save and Load Test" begin
        input_size = 3
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        randomize_diffusion_model!(model)

        save_path = joinpath(@__DIR__(), "test_dm_model.h5")
        test_model_save(save_path, model, input_size)
        
        loaded_protocol = GenerativeModelProtocol(save_path)
        loaded_model = loaded_protocol.model
        test_compare_models(model, loaded_model)

        model_β = loaded_model.β
        model_denoiser = GenerativeModelProtocols.TabularDenoiser(save_path)
        loaded_model = GenerativeModelProtocols.DiffusionModel(model_β, model_denoiser)
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

