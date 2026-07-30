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
GenerativeModelProtocols.TabularDenoiser
```

# Methods
```@docs
train!(::GenerativeModelProtocol)
categorize(::GenerativeModelProtocol, ::Vector{Float64})
categorize(::GenerativeModelProtocol, ::Matrix{Float64})
GenerativeModelProtocols.encode(::GenerativeModelProtocols.VariationalAutoencoder, ::Vector{Float64})
GenerativeModelProtocols.encode(::GenerativeModelProtocols.VariationalAutoencoder, ::Matrix{Float64})
GenerativeModelProtocols.decode(::GenerativeModelProtocols.VariationalAutoencoder, ::Vector{Float64})
GenerativeModelProtocols.decode(::GenerativeModelProtocols.VariationalAutoencoder, ::Matrix{Float64})
```

# Convenience Constructors
```@docs
GenerativeModelProtocols.GenerativeModelProtocol(::GenerativeModelProtocols.AbstractGenerativeModel, ::Matrix{Float64})
GenerativeModelProtocols.VariationalAutoencoder(::Int, ::Int, ::Int)
GenerativeModelProtocols.VariationalAutoencoder(::Tuple{Vararg{Chain}}, ::Tuple{Vararg{Chain}})
GenerativeModelProtocols.GaussianMixtureModel(::Int, ::Int)
GenerativeModelProtocols.DiffusionModel(::Vector{Float64}, ::GenerativeModelProtocols.TabularDenoiser)
GenerativeModelProtocols.DiffusionModel(::Int, ::Vector{Float64})
GenerativeModelProtocols.DiffusionModel(::Int, ::Float64, ::Float64, ::GenerativeModelProtocols.TabularDenoiser)
GenerativeModelProtocols.DiffusionModel(::Int, ::Int, ::Float64, ::Float64)
GenerativeModelProtocols.TabularDenoiser(::Int; ::Int, ::Int, ::Function, ::Float64)
```

