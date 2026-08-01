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
categorize(::GenerativeModelProtocol, ::Vector)
categorize(::GenerativeModelProtocol, ::Matrix)
categorize(::GenerativeModelProtocols.GaussianMixtureModel, ::Vector)
categorize(::GenerativeModelProtocols.GaussianMixtureModel, ::Matrix)
encode(::GenerativeModelProtocol, ::Vector)
encode(::GenerativeModelProtocol, ::Matrix)
decode(::GenerativeModelProtocol, ::Vector)
decode(::GenerativeModelProtocol, ::Matrix)
encode(::GenerativeModelProtocols.VariationalAutoencoder, ::Vector)
encode(::GenerativeModelProtocols.VariationalAutoencoder, ::Matrix)
decode(::GenerativeModelProtocols.VariationalAutoencoder, ::Vector)
decode(::GenerativeModelProtocols.VariationalAutoencoder, ::Matrix)
encode(::GenerativeModelProtocols.DiffusionModel, ::Vector)
encode(::GenerativeModelProtocols.DiffusionModel, ::Matrix)
decode(::GenerativeModelProtocols.DiffusionModel, ::Vector)
decode(::GenerativeModelProtocols.DiffusionModel, ::Matrix)
```

# Convenience Constructors
```@docs
GenerativeModelProtocols.GenerativeModelProtocol(::GenerativeModelProtocols.AbstractGenerativeModel, ::Matrix{Float64})
GenerativeModelProtocols.VariationalAutoencoder(::Int, ::Int, ::Int)
GenerativeModelProtocols.VariationalAutoencoder(::Tuple{Vararg{Chain}}, ::Tuple{Vararg{Chain}})
GenerativeModelProtocols.GaussianMixtureModel(::Int, ::Int)
GenerativeModelProtocols.DiffusionModel(::Tuple{Float64}, ::GenerativeModelProtocols.TabularDenoiser)
GenerativeModelProtocols.DiffusionModel(::Int, ::Tuple{Float64})
GenerativeModelProtocols.DiffusionModel(::Int, ::Float64, ::Float64, ::GenerativeModelProtocols.TabularDenoiser)
GenerativeModelProtocols.DiffusionModel(::Int, ::Int, ::Float64, ::Float64)
GenerativeModelProtocols.TabularDenoiser(::Int; ::Int, ::Int, ::Function, ::Float64)
```

