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
GenerativeModelProtocols.GaussianMixtureModel
```

# Methods
```@docs
train!(::GenerativeModelProtocol)
categorize(::GenerativeModelProtocol, ::Vector{Float64})
categorize(::GenerativeModelProtocol, ::Matrix{Float64})
GenerativeModelProtocols.encode(::GenerativeModelProtocols.VariationalAutoencoder, ::Any)
GenerativeModelProtocols.decode(::GenerativeModelProtocols.VariationalAutoencoder, ::Any)
```

# Convenience Constructors
```@docs
GenerativeModelProtocols.GenerativeModelProtocol(::GenerativeModelProtocols.AbstractGenerativeModel, ::Matrix{Float64})
GenerativeModelProtocols.VariationalAutoencoder(::Int, ::Int, ::Int)
GenerativeModelProtocols.VariationalAutoencoder(::Tuple{Vararg{Chain}}, ::Tuple{Vararg{Chain}})
GenerativeModelProtocols.GaussianMixtureModel(::Int, ::Int)
```
