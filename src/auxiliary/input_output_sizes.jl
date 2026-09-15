function _input_size(layer::L) where {L<:Dense}
    return layer.in_dims
end

function _input_size(chain::C) where {C<:Chain}
    return _input_size(chain[1])
end

function _input_size(layer::NamedTuple)
    if isempty(layer) || isempty(layer[1])
        return 0
    end

    return size(layer[1].weight, 2)
end

function _output_size(layer::L) where {L<:Dense}
    return layer.out_dims
end

function _output_size(chain::C) where {C<:Chain}
    return _output_size(chain[end])
end

function _output_size(layer::NamedTuple)
    if isempty(layer) || isempty(layer[1])
        return 0
    end

    return size(layer[end].weight, 1)
end

function _is_empty_state_tree(nt::NamedTuple)
    if isempty(nt)
        return true
    end
    
    return all(val -> val isa NamedTuple && _is_empty_state_tree(val), values(nt))
end
