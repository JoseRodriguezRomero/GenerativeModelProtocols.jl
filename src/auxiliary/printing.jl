macro _default_print_padding()
    return "   "
end

function _base_print_layers(layers::Tuple{Vararg{<:Dense}}, print_padding::String = @_default_print_padding)
    for layer in layers
        print(print_padding * print_padding)
        println(layer)
    end
end

function _print_chains(chains::Tuple{Vararg{<:Chain}}, print_padding::String = @_default_print_padding)
    for i in eachindex(chains)
        println(print_padding * "Chain(")
        _base_print_layers(chains[i].layers)
        println(print_padding * ")")
    end
end

function _print_layers(layers::Tuple{Vararg{<:Dense}}, print_padding::String = @_default_print_padding)
    println(print_padding * "Tuple()")
    _base_print_layers(Tuple(layers))
    println(print_padding * ")")
end

function _print_chains(chain::C, print_padding) where {C<:Chain}
    _print_chains((chain,), print_padding)
end

