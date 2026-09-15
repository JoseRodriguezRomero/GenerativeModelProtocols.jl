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
GenerativeModelProtocols.GenerativeAdversarialNetwork
GenerativeModelProtocols.NormalizingFlow
```

# Methods
```@docs
train!(::GenerativeModelProtocol)
categorize(::GenerativeModelProtocol, ::Vector)
categorize(::GenerativeModelProtocol, ::Matrix)
encode(::GenerativeModelProtocol, ::Vector)
encode(::GenerativeModelProtocol, ::Matrix)
decode(::GenerativeModelProtocol, ::Vector)
decode(::GenerativeModelProtocol, ::Matrix)
input_size(::GenerativeModelProtocol)
latent_size(::GenerativeModelProtocol)
```

# Convenience Constructors
```@docs
GenerativeModelProtocols.GenerativeModelProtocol(::GenerativeModelProtocols.AbstractGenerativeModel, ::Matrix{<:AbstractFloat})
GenerativeModelProtocols.VariationalAutoencoder(::Int, ::Int, ::Int)
GenerativeModelProtocols.GaussianMixtureModel(::Int, ::Int)
GenerativeModelProtocols.DiffusionModel(::Int, ::Tuple{Vararg{AbstractFloat}})
GenerativeModelProtocols.DiffusionModel(::Int, ::F, ::F, ::GenerativeModelProtocols.TabularDenoiser{LayerNames, F}) where {LayerNames, F<:AbstractFloat}
GenerativeModelProtocols.DiffusionModel(::Int, ::Int, ::F, ::F) where {F<:AbstractFloat}
GenerativeModelProtocols.TabularDenoiser(::Int; ::Int, ::Int, ::Function, <:AbstractFloat)
GenerativeModelProtocols.GenerativeAdversarialNetwork(::Int, ::Int)
GenerativeModelProtocols.NormalizingFlow(::Int)
```

