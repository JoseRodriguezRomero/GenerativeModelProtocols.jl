macro _nf_default_activation_function()
    return swish
end

function default_velocity_field(input_size::Int, hidden_layer_size::Int = 32, activation_function::Function = @_nf_default_activation_function)
    return Chain(
        Dense((input_size + 1) => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => hidden_layer_size, activation_function),
        Dense(hidden_layer_size => input_size)
    )
end

function compatible_nf_model(velocity_field::C) where {C <: Chain}
    if _output_size(velocity_field) != (_input_size(velocity_field) - 1)
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
@kwdef struct NormalizingFlow <: AbstractGenerativeModel
    """Neural network parametrizing the velocity field as a function of time."""
    velocity_field::Chain
    """Trained parameters of the model. Users should not use this directly."""
    _ps::Union{Ref{<:NamedTuple}, Nothing} = nothing
    """Trained state of the model. Users should not use this directly."""
    _st::Union{Ref{<:NamedTuple}, Nothing} = nothing
    
    function NormalizingFlow(velocity_field::Chain, _ps::Union{Ref{<:NamedTuple}, Nothing}, _st::Union{Ref{<:NamedTuple}, Nothing})
        if !compatible_nf_model(velocity_field)
            @error "Incompatible NormalizingFlow architecture!"
            throw(MethodError(NormalizingFlow, (velocity_field)))
        end

        if isnothing(_ps) && isnothing(_st)
            _ps_val, _st_val = Lux.setup(Random.default_rng(), (velocity_field = velocity_field,))
            _ps = Ref{NamedTuple}(_ps_val)
            _st = Ref{NamedTuple}(_st_val)
        end

        return new(velocity_field, _ps, _st)
    end
end

"""
    GenerativeModelProtocols.NormalizingFlow(input_size::Int)

Convenience constructor that creates a 
`GenerativeModelProtocols.NormalizingFlow` using default velocity field network 
architecture.
"""
function NormalizingFlow(input_size::Int)
    return NormalizingFlow(;
        velocity_field = default_velocity_field(input_size)
    )
end

function _train!(protocol::GenerativeModelProtocol, model::NormalizingFlow; print_log::Bool = true)
    training_data_device = protocol.training_data |> protocol.device
    loader = load_data(training_data_device, protocol.batchsize, protocol.shuffle)

    ps = protocol.precision(model._ps[]) |> protocol.device
    st = protocol.precision(model._st[]) |> protocol.device

    FP = eltype(protocol.training_data)

    function _nf_train_step!(x, p_current, s_current, o_current)
        x₀ = randn_like(x, size(x))
        x₁ = x
        t = rand_like(x₁, (1, size(x₀,2)))
        xₜ = (FP(1.0) .- t) .* x₀ + t .* x₁

        vₜ = x₁ - x₀

        _objective = (p) -> begin
            ps_vf = p.velocity_field
            st_vf = s_current.velocity_field

            v̂ₜ, _ = model.velocity_field(vcat(xₜ, t), ps_vf, st_vf)
            return mean(abs2, v̂ₜ - vₜ)
        end

        loss_val = _objective(p_current)
        loss_grads = Enzyme.make_zero(p_current)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(_objective),
            Enzyme.Active,
            Enzyme.Duplicated(p_current, loss_grads)
        )

        o_updated, p_updated = Optimisers.update(o_current, p_current, loss_grads)

        return loss_val, p_updated, o_updated
    end

    opt_state = _initial_step(model, ps, st, protocol.optimiser)
    _train_step!, opt_state = _train_step_device_dispatch(protocol.device, _nf_train_step!, loader, opt_state)

    if print_log; println("Training Diffusion Model...") end
    for epoch in 1:protocol.epochs
        epoch_loss = FP(0.0)

        if epoch % 100 == 0 && protocol.shuffle
            loader = load_data(training_data_device, protocol.batchsize, protocol.shuffle)
        end

        for x_batch in loader
            mse_loss, opt_state = _train_step!(x_batch, opt_state)
            epoch_loss += mse_loss
        end

        protocol._log.loss[epoch] = epoch_loss / length(loader)

        if epoch % 5 == 0 || epoch == 1
            average_loss = protocol._log.loss[epoch]
            if print_log
                @printf("Epoch %8d | Avg. MSE: %16.8e \n", epoch, average_loss)
            end
        end
    end
    if print_log; println("Training complete!") end

    model._ps[] = opt_state.parameters
    model._st[] = opt_state.states

    return protocol._log
end

function _normalizing_flow_ode_solve_forward end
function _normalizing_flow_ode_solve_reverse end

function _latent_size(model::NormalizingFlow)
    return size(model._ps[].velocity_field[end].weight,1)
end

function _encode(model::NormalizingFlow, x::Matrix; ode_solver = nothing)
    return _normalizing_flow_ode_solve_reverse(x, model, ode_solver)
end

function _encode(model::NormalizingFlow, x::Vector; ode_solver = nothing)
    return _encode(model, reshape(x, :, 1); ode_solver = ode_solver)[:]
end

function _decode(model::NormalizingFlow, z::Matrix; ode_solver = nothing)
    return _normalizing_flow_ode_solve_forward(z, model, ode_solver)
end

function _decode(model::NormalizingFlow, z::Vector; ode_solver = nothing)
    return _decode(model, reshape(z,:,1); ode_solver = ode_solver)[:]
end

function _eval(model::NormalizingFlow, n_samples::Int; ode_solver = nothing)
    x₀ = randn_like(model._ps[].velocity_field.layer_1.weight, (_latent_size(model), n_samples))
    return _normalizing_flow_ode_solve_forward(x₀, model, ode_solver)
end

function _eval(model::NormalizingFlow; ode_solver = nothing)
    return _eval(model,1; ode_solver = ode_solver)[:]
end

