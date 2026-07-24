# [Reference](@id reference)

## Contents

```@contents
Pages = ["api.md"]
```

## Index
```@index
Pages = ["api.md"]
```

# Types
```@docs
GenerativeModelProtocol
GenerativeModelProtocols.VariationalAutoencoder
GenerativeModelProtocols.DiffusionModel
```

# Methods
```@docs
GenerativeModelProtocols.train!(::GenerativeModelProtocol)
GenerativeModelProtocols.encode(::GenerativeModelProtocols.VariationalAutoencoder, ::Any)
GenerativeModelProtocols.decode(::GenerativeModelProtocols.VariationalAutoencoder, ::Any)
```

# Convenience Constructors
```@docs
GenerativeModelProtocols.GenerativeModelProtocol(::GenerativeModelProtocols.AbstractGenerativeModel, ::Matrix{Float64})
GenerativeModelProtocols.VariationalAutoencoder(::Int, ::Int, ::Int)
GenerativeModelProtocols.VariationalAutoencoder(::Tuple{Vararg{Chain}}, ::Tuple{Vararg{Chain}})
```
