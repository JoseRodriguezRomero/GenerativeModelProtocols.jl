function _input_size(layer::L) where {L<:Dense}
    return size(layer.weight, 2)
end

function _input_size(chain::C) where {C<:Chain}
    return _input_size(chain[1])
end

function _output_size(layer::L) where {L<:Dense}
    return size(layer.weight, 1)
end

function _output_size(chain::C) where {C<:Chain}
    return _output_size(chain[end])
end

