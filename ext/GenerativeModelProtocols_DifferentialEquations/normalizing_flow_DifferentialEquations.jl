macro _default_ode_solver()
    return DifferentialEquations.Tsit5()
end

function GenerativeModelProtocols._normalizing_flow_ode_solve(x₀::Matrix{<:AbstractFloat}, velocity_field, ode_solver)
    if isnothing(ode_solver)
        ode_solver = @_default_ode_solver
    end

    T = eltype(x₀)

    function _ode_problem!(du, u, p, t)
        t_vec = zeros(T, 1, (size(u,2))) .+ T(t)
        du[1:(end-1), :] = velocity_field(vcat(u, t_vec))
    end

    tspan = (T(0.0), T(1.0))
    prob = ODEProblem(_ode_problem!, x₀, tspan)
    sol = solve(prob, ode_solver)
end

