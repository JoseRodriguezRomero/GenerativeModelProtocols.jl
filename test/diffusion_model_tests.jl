function test_compare_tabular_denoiser(denoiser_a::GenerativeModelProtocols.TabularDenoiser, denoiser_b::GenerativeModelProtocols.TabularDenoiser)
    ϵ = 1.0E-9

    @test compare_chains(denoiser_a._ps[].time_embedding_mlp, denoiser_b._ps[].time_embedding_mlp)
    @test compare_layers(denoiser_a._ps[].input_projection, denoiser_b._ps[].input_projection)
    @test compare_tuple_layers(denoiser_a._ps[].residual_layers, denoiser_b._ps[].residual_layers)
    @test compare_tuple_layers(denoiser_a._ps[].time_projection_layers, denoiser_b._ps[].time_projection_layers)
    @test compare_layers(denoiser_a._ps[].output_projection, denoiser_b._ps[].output_projection)
    @test abs(denoiser_a.max_period - denoiser_b.max_period) < ϵ
end

function test_compare_models(model_a::GenerativeModelProtocols.DiffusionModel, model_b::GenerativeModelProtocols.DiffusionModel)
    ϵ = 1.0E-9

    @test maximum(collect(model_a.β) - collect(model_b.β)) < ϵ
    test_compare_tabular_denoiser(model_a.denoiser_model, model_b.denoiser_model)
end

function randomize_tabular_denoiser!(denoiser::GenerativeModelProtocols.TabularDenoiser)
    randomize_chain!(denoiser._ps[].time_embedding_mlp)
    randomize_layer!(denoiser._ps[].input_projection)
    randomize_layers!(denoiser._ps[].residual_layers)
    randomize_layers!(denoiser._ps[].time_projection_layers)
    randomize_layer!(denoiser._ps[].output_projection)
end

function example_diffusion_model(; 
    T::Union{Int64, Nothing} = nothing,
    β::Union{Tuple{Vararg{Float32}}, Nothing} = nothing)

    default_model = GenerativeModelProtocols.DiffusionModel(2, 30)

    if isnothing(β)
        β = default_model.β
    end

    if !isnothing(T)
        default_model = GenerativeModelProtocols.DiffusionModel(2, T)
    end

    return GenerativeModelProtocols.DiffusionModel(;
        β              = β,
        denoiser_model = default_model.denoiser_model
    )
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
        test_train_model_no_data(model, input_size)
    end

    @testset "Incompatible DiffusionModel Architecture Test" begin
        default_dm = example_diffusion_model()
        default_T = default_dm.denoiser_model.T

        # Test incompatible vector lengths
        @test_throws Exception example_diffusion_model(; β = Tuple(rand(Float32, default_T + 1)))
    end

    @testset "Make Synthetic Data Test" begin
        input_size = 2
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        test_model_make_synthetic_data(model, input_size)
    end

    @testset "Encode/Decode Test" begin
        input_size = 3
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
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
        @test size(z) == (input_size, size(x,2))
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
        
        loaded_model = GenerativeModelProtocols.DiffusionModel(save_path)
        test_compare_models(model, loaded_model)

        model_β = loaded_model.β
        model_denoiser = GenerativeModelProtocols.TabularDenoiser(save_path)
        loaded_model = GenerativeModelProtocols.DiffusionModel(model_β, model_denoiser)
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

