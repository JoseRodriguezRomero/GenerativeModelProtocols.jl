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
    @test maximum(collect(model_a.α) - collect(model_b.α)) < ϵ
    @test maximum(collect(model_a.ᾱ) - collect(model_b.ᾱ)) < ϵ
    @test maximum(collect(model_a.β) - collect(model_b.β)) < ϵ
    test_compare_tabular_denoiser(model_a.denoiser_model, model_b.denoiser_model)
end

function randomize_tabular_denoiser!(denoiser::GenerativeModelProtocols.TabularDenoiser)
    randomize_chains!(denoiser.time_embedding_mlp)
    randomize_layers!(denoiser.input_projection)
    randomize_layers!(denoiser.residual_layers)
    randomize_layers!(denoiser.time_projection_layers)
    randomize_layers!(denoiser.output_projection)
end

function example_diffusion_model(; 
    T::Union{Int64, Nothing} = nothing, 
    α::Union{Tuple{Vararg{Float64}}, Nothing} = nothing,
    ᾱ::Union{Tuple{Vararg{Float64}}, Nothing} = nothing,
    β::Union{Tuple{Vararg{Float64}}, Nothing} = nothing)

    default_model = GenerativeModelProtocols.DiffusionModel(2, 30)

    if isnothing(T)
        T = default_model.T
    end

    if isnothing(α)
        α = default_model.α
    end

    if isnothing(ᾱ)
        ᾱ = default_model.ᾱ
    end

    if isnothing(β)
        β = default_model.β
    end

    return GenerativeModelProtocols.DiffusionModel(;
        T              = T,
        α              = α,
        ᾱ              = ᾱ,
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
        test_train_model_no_data(model)
    end

    @testset "Incompatible VariationalAutoencoder Architecture Test" begin
        default_dm = example_diffusion_model()

        # Test incompatible T
        @test_throws Exception example_diffusion_model(; T = default_dm.T + 1)

        # Test incompatible vector lengths
        @test_throws Exception example_diffusion_model(; α = Tuple(rand(Float64, default_dm.T + 1)))
        @test_throws Exception example_diffusion_model(; ᾱ = Tuple(rand(Float64, default_dm.T + 1)))
        @test_throws Exception example_diffusion_model(; β = Tuple(rand(Float64, default_dm.T + 1)))

        # Test incompatible α
        @test_throws Exception example_diffusion_model(; α = Tuple(rand(Float64, default_dm.T)))
        @test_throws Exception example_diffusion_model(; ᾱ = Tuple(rand(Float64, default_dm.T)))
    end

    @testset "Make Synthetic Data Test" begin
        input_size = 2
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
        test_model_make_synthetic_data(model)
    end

    @testset "Encode/Decode Test" begin
        input_size = 3
        T = 30

        model = GenerativeModelProtocols.DiffusionModel(input_size, T)
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
        @test size(z) == (input_size, size(x,2))
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

