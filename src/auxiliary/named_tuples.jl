function _named_tuples_from_tuple(src::Tuple, base_symbol::String)
    names = ntuple(i -> Symbol(base_symbol, "_", i), length(src))
    return NamedTuple{names}(src)
end

