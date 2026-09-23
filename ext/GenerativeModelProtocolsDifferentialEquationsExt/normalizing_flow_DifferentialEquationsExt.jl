macro _default_ode_solver()
    return DifferentialEquations.Tsit5()
end

function _normalizing_flow_ode_solve(x₀::Matrix{<:AbstractFloat}, tspan, model, ode_solver)
    if isnothing(ode_solver)
        ode_solver = @_default_ode_solver
    end

    T = eltype(x₀)

    velocity_field = model.velocity_field
    ps = model._ps[].velocity_field
    st = model._st[].velocity_field

    function _ode_problem!(du, u, p, t)
        t_vec = fill(t, 1, size(u, 2))
        du .= first(velocity_field(vcat(u, t_vec), ps, st))
    end

    prob = ODEProblem(_ode_problem!, x₀, tspan)
    sol = solve(prob, ode_solver)

    return sol(tspan[end])
end

function GenerativeModelProtocols._normalizing_flow_ode_solve_forward(x₀::Matrix{<:AbstractFloat}, model, ode_solver, t_final::AbstractFloat = 1.0)
    T = eltype(x₀)
    tspan = (T(0.0), T(t_final))

    return _normalizing_flow_ode_solve(x₀, tspan, model, ode_solver)
end

function GenerativeModelProtocols._normalizing_flow_ode_solve_reverse(x₁::Matrix{<:AbstractFloat}, model, ode_solver, t_final::AbstractFloat = 0.0)
    T = eltype(x₁)
    tspan = (T(1.0), T(t_final))

    return _normalizing_flow_ode_solve(x₁, tspan, model, ode_solver)
end

