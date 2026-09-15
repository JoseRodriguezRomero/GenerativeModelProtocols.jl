function _initial_step(model, ps, st, optimiser)
    return Lux.Training.TrainState(
        nothing,                         # cache
        nothing,                         # objective_function
        nothing,                         # allocator_cache
        model,                           # model struct
        ps,                              # parameters NamedTuple
        st,                              # states NamedTuple
        optimiser,                       # optimiser
        Optimisers.setup(optimiser, ps), # optimizer state
        0                                # step count
    )
end

function _function_device_dispatch(::Lux.CPUDevice, _train_function::Function, args...)
    return _train_function
end

function _function_device_dispatch(::Lux.ReactantDevice, _train_function::Function, args...)
    return @compile _train_function(args...)
end

function _train_step_device_dispatch(::Lux.CPUDevice, train_step_func!::Function, loader, opt_state)
    first_batch = first(loader)

    p_init = opt_state.parameters
    s_init = opt_state.states
    o_init = opt_state.optimizer_state

    _, p_init, o_init = train_step_func!(first_batch, p_init, s_init, o_init)
    
    opt_state = Lux.Training.TrainState(
        opt_state.cache, opt_state.objective_function, opt_state.allocator_cache,
        opt_state.model, p_init, s_init, opt_state.optimizer, o_init, opt_state.step
    )

    _train_step! = (x, state) -> begin
        res_loss, p_up, o_up = train_step_func!(x, state.parameters, state.states, state.optimizer_state)
        state_up = Lux.Training.TrainState(
            state.cache, state.objective_function, state.allocator_cache,
            state.model, p_up, state.states, state.optimizer, o_up, state.step + 1
        )
        return res_loss, state_up
    end

    return _train_step!, opt_state
end

function _train_step_device_dispatch(::Lux.ReactantDevice, train_step_func!::Function, loader, opt_state)
    first_batch = first(loader)

    p_init = opt_state.parameters
    s_init = opt_state.states
    o_init = opt_state.optimizer_state

    _, p_init, o_init = @jit train_step_func!(first_batch, p_init, s_init, o_init)
    
    opt_state = Lux.Training.TrainState(
        opt_state.cache, opt_state.objective_function, opt_state.allocator_cache,
        opt_state.model, p_init, s_init, opt_state.optimizer, o_init, opt_state.step
    )

    _compiled_pass = @compile train_step_func!(
        first_batch, opt_state.parameters, opt_state.states, opt_state.optimizer_state
    )

    _train_step! = (x, state) -> begin
        if isa(x, Tuple)
            if size(x[1], 2) != size(first_batch[1], 2)
                T = eltype(x[1])
                return T(0.0), state
            end
        else
            if size(x, 2) != size(first_batch, 2)
                T = eltype(x)
                return T(0.0), state
            end
        end

        res_loss, p_up, o_up = _compiled_pass(
            x, state.parameters, state.states, state.optimizer_state
        )
        
        state_up = Lux.Training.TrainState(
            state.cache, state.objective_function, state.allocator_cache,
            state.model, p_up, state.states, state.optimizer, o_up, state.step + 1
        )

        return res_loss, state_up
    end

    return _train_step!, opt_state
end

