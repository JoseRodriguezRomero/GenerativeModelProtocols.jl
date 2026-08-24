macro _nf_default_activation_function()
    return swish
end

function default_velocity_field(input_size::Int, activation_function::Function = @_nf_default_activation_function)
    return Chain(
        Dense(input_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => (input_size + 1))
    )
end

function compatible_nf_model(velocity_field::C) where {C <: Chain}
    if _output_size(velocity_field) != (_input_size(velocity_field) + 1)
        return false
    end

    return true
end

"""
$TYPEDEF

A structure containing the general parameters needed to evaluate and train a 
Continuous Normalizing Flow (CNF). Once trained, it can be used a generative 
model.

$TYPEDFIELDS
"""
@kwdef struct NormalizingFlow <: AbstractCategoricalGenerativeModel
    """Neural network parametrizing the velocity field as a function of time."""
    velocity_field::Chain
    
    function GenerativeAdversarialNetwork(velocity_field::C) where {C<:Chain}
        if !compatible_nf_model(velocity_field)
            @error "Incompatible NormalizingFlow architecture!"
            throw(MethodError(NormalizingFlow, (velocity_field)))
        end

        return new(velocity_field)
    end
end

"""
    GenerativeModelProtocols.NormalizingFlow(input_size::Int)

Convenience constructor that creates a 
`GenerativeModelProtocols.NormalizingFlow` using default velocity field network 
architecture.
"""
function NormalizingFlow(input_size::Int)
    return NormalizingFlow(default_velocity_field(input_size))
end

function _train!(protocol::GenerativeModelProtocol, model::NormalizingFlow; print_log::Bool = true)
    model_train_device = model |> protocol.device
    opt_state = Flux.setup(protocol.optimiser, model_train_device)
    
    batchsize_device = protocol.batchsize |> protocol.device
    training_data_device = protocol.training_data |> protocol.device

    loader = Flux.DataLoader(
        training_data_device, 
        batchsize = batchsize_device, 
        shuffle = false,
        parallel = true
    )
    loader_length = length(loader)
    loader_length_device = loader_length |> protocol.device

    FP = eltype(protocol.precision([1.0]))

    if print_log; println("Training Diffusion Model...") end
    for epoch in 1:protocol.epochs
        epoch_loss = 0.0
        total_grad_norm = 0.0

        if epoch % 100 == 0 && protocol.shuffle
            loader = Flux.DataLoader(
                shuffleobs(training_data_device), 
                batchsize = batchsize_device, 
                shuffle = false,
                parallel = true
            )
        end

        for x_batch in loader
            loss, (grads,) = Flux.withgradient(AutoEnzyme(), model_train_device) do model
                x₀ = Flux.randn_like(x_batch, size(x_batch))
                x₁ = x_batch
                t = Flux.rand_like(x₁, size(x₀,2))
                xₜ = (FP(1.0) .- t) .* x₀ + t .* x₁

                vₜ = x₁ - x₀
                v̂ₜ = model.velocity_field(vcat(xₜ, t))
                return Flux.Losses.mse(v̂ₜ, vₜ)
            end
            
            raw_gradient_arrays = Optimisers.trainables(grads)
            batch_grad_norm = sqrt(sum(sum(abs2, g) for g in raw_gradient_arrays if g isa AbstractArray))

            Flux.update!(opt_state, model_train_device, grads)

            epoch_loss += loss / loader_length_device
            total_grad_norm += batch_grad_norm / loader_length_device
        end

        protocol._log.loss[epoch] = epoch_loss
        protocol._log.loss_grad_norm[epoch] = total_grad_norm

        if (epoch % 5 == 0 || epoch == 1) && print_log
            @printf("Epoch: %8d | Mean MSE Loss: %-15.8f | Grad Norm: %-10.8f\n", epoch, protocol._log.loss[epoch], protocol._log.loss_grad_norm[epoch])
        end
    end

    Flux.loadmodel!(model.velocity_field, model_train_device.velocity_field)

    return protocol._log
end

