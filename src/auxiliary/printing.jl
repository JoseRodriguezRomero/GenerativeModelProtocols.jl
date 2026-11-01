macro _default_print_padding()
    return "   "
end

function _base_print_layers(layers::NamedTuple{Names, <:Tuple{Vararg{Dense}}}, print_padding::String = @_default_print_padding) where {Names}
    for key in keys(layers)
        print(print_padding * print_padding)
        println(layers[key])
    end
end

function _print_chains(chains::NamedTuple{Names, <:Tuple{Vararg{Chain}}}, print_padding::String = @_default_print_padding) where {Names}
    for i in eachindex(chains)
        println(print_padding * "Chain(")
        _base_print_layers(chains[i].layers)
        println(print_padding * ")")
    end
end

function _print_layers(layers::NamedTuple{Names, <:Tuple{Vararg{Dense}}}, print_padding::String = @_default_print_padding) where {Names}
    println(print_padding * "Tuple()")
    _base_print_layers(layers)
    println(print_padding * ")")
end

function _print_chains(chain::C, print_padding::String = @_default_print_padding) where {C<:Chain}
    _print_chains((chain_1 = chain,), print_padding)
end

function _print_trace_iter(iter, max_iters; kwargs...)
    epoch_width = max(5, length(string(max_iters)))
    print("Epoch", " "^max(3, epoch_width - length("Epoch") + 1), lpad(iter, epoch_width))
    for (_, (name, value)) in pairs(kwargs)
        print(" | ", name, ": ", value isa AbstractFloat ? @sprintf("%16.8E", value) : value)
    end
    println()
end

