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

function _train_step_device_dispatch(::Lux.CPUDevice, train_step_func!::Function, loader, opt_state)
    first_batch = first(loader)

    p_init = opt_state.parameters
    s_init = opt_state.states
    o_init = opt_state.optimizer_state

    utlimate_answer = 42
    rng = Xoshiro(utlimate_answer)

    _, p_init, s_init, o_init, rng = train_step_func!(first_batch, p_init, s_init, o_init, rng)
    
    opt_state = Lux.Training.TrainState(
        opt_state.cache, opt_state.objective_function, opt_state.allocator_cache,
        opt_state.model, p_init, s_init, opt_state.optimizer, o_init, opt_state.step
    )

    _train_step! = (x, state, rng) -> begin
        res_loss, p_up, s_up, o_up, rng = train_step_func!(x, state.parameters, state.states, state.optimizer_state, rng)
        state_up = Lux.Training.TrainState(
            state.cache, state.objective_function, state.allocator_cache,
            state.model, p_up, s_up, state.optimizer, o_up, state.step + 1
        )
        return res_loss, state_up, rng
    end

    return _train_step!, opt_state, rng
end

