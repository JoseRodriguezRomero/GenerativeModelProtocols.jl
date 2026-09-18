function randomize_normalizing_flow!(model::GenerativeModelProtocols.NormalizingFlow)
    randomize_chain!(model._ps[].velocity_field)
end

function test_compare_models(model_a::GenerativeModelProtocols.NormalizingFlow, model_b::GenerativeModelProtocols.NormalizingFlow)
    @test compare_chains(model_a._ps[].velocity_field, model_b._ps[].velocity_field)
end

@testset "NormalizingFlow Tests" begin
    @testset "Constructors Tests" begin
        input_size = 3

        model = GenerativeModelProtocols.NormalizingFlow(input_size)
        @test isa(model, GenerativeModelProtocols.NormalizingFlow)
    end

    @testset "Train Test" begin
        train_data = make_test_train_data(50)

        input_size = size(train_data,1)

        model = GenerativeModelProtocols.NormalizingFlow(input_size)
        test_train_model(model, train_data)
        test_train_model_no_data(model, input_size)
        # test_train_model_reactant(model, train_data)
    end

    @testset "Incompatible NormalizingFlow Architecture Test" begin
        input_size = 3
        hidden_layer_size = 16
        activation_function = relu

        velocity_field = Chain(
            Dense(input_size => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => hidden_layer_size, activation_function),
            Dense(hidden_layer_size => input_size)
        )

        @test_throws Exception GenerativeModelProtocols.NormalizingFlow(; velocity_field = velocity_field)
    end

    @testset "Make Synthetic Data Test" begin
        input_size = 2

        model = GenerativeModelProtocols.NormalizingFlow(input_size)
        test_model_make_synthetic_data(model, input_size)
    end

    @testset "Encode/Decode Test" begin
        input_size = 3

        model = GenerativeModelProtocols.NormalizingFlow(input_size)
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

        model = GenerativeModelProtocols.NormalizingFlow(input_size)
        test_model_display_no_data(model, input_size)
        test_model_display(model, input_size)
    end

    @testset "Save and Load Test" begin
        input_size = 3

        model = GenerativeModelProtocols.NormalizingFlow(input_size)
        randomize_normalizing_flow!(model)

        save_path = joinpath(@__DIR__(), "test_nf_model.h5")
        test_model_save(save_path, model, input_size)
        
        loaded_model = GenerativeModelProtocols.NormalizingFlow(save_path)
        test_compare_models(model, loaded_model)

        remove_file(save_path)
    end
end

