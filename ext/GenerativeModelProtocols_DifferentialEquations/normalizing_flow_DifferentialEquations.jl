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
        t_vec = zeros(T, 1, size(u, 2)) .+ T(t)
        du .= first(velocity_field(vcat(u, t_vec), ps, st))
    end

    prob = ODEProblem(_ode_problem!, x₀, tspan)
    sol = solve(prob, ode_solver)

    return sol.u[end]
end

function GenerativeModelProtocols._normalizing_flow_ode_solve_forward(x₀::Matrix{<:AbstractFloat}, model, ode_solver)
    T = eltype(x₀)
    tspan = (T(0.0), T(1.0))

    return _normalizing_flow_ode_solve(x₀, tspan, model, ode_solver)
end

function GenerativeModelProtocols._normalizing_flow_ode_solve_reverse(x₁::Matrix{<:AbstractFloat}, model, ode_solver)
    T = eltype(x₁)
    tspan = (T(1.0), T(0.0))

    return _normalizing_flow_ode_solve(x₁, tspan, model, ode_solver)
end

