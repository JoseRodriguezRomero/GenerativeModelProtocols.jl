module GenerativeModelProtocolsReactantExt

import GenerativeModelProtocols
import Lux, Random

using Reactant

function GenerativeModelProtocols._function_device_dispatch(::Lux.ReactantDevice, _train_function::Function, args...)
    return @compile _train_function(args...)
end

function GenerativeModelProtocols._train_step_device_dispatch(::Lux.ReactantDevice, train_step_func!::Function, loader, opt_state)
    first_batch = first(loader)

    p_init = opt_state.parameters
    s_init = opt_state.states
    o_init = opt_state.optimizer_state
    rng = Reactant.ReactantRNG()

    _, p_init, s_init, o_init, rng = @jit train_step_func!(first_batch, p_init, s_init, o_init, rng)
    _compiled_pass = @compile train_step_func!(first_batch, p_init, s_init, o_init, rng)

    opt_state = Lux.Training.TrainState(
        opt_state.cache, opt_state.objective_function, opt_state.allocator_cache,
        opt_state.model, p_init, s_init, opt_state.optimizer, o_init, opt_state.step
    )

    _train_step! = (x, state, rng) -> begin
        if isa(x, Tuple)
            if size(x[1], 2) != size(first_batch[1], 2)
                T = eltype(x[1])
                return T(0.0), state, rng
            end
        else
            if size(x, 2) != size(first_batch, 2)
                T = eltype(x)
                return T(0.0), state, rng
            end
        end

        res_loss, p_up, s_up, o_up, rng = _compiled_pass(
            x, state.parameters, state.states, state.optimizer_state, rng
        )
        
        state_up = Lux.Training.TrainState(
            state.cache, state.objective_function, state.allocator_cache,
            state.model, p_up, s_up, state.optimizer, o_up, state.step + 1
        )

        return res_loss, state_up, rng
    end

    return _train_step!, opt_state, rng
end

end

