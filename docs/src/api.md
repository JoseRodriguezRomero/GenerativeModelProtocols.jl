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
GenerativeModelProtocols.default_optimiser(::GenerativeModelProtocol)
GenerativeModelProtocols.sinkhorn_distance(::AbstractMatrix, ::AbstractMatrix)
GenerativeModelProtocols.sinkhorn_distance(::GenerativeModelProtocol, ::AbstractMatrix)
GenerativeModelProtocols.sinkhorn_distance(::GenerativeModelProtocol)
GenerativeModelProtocols.earth_mover_distance(::AbstractMatrix, ::AbstractMatrix)
GenerativeModelProtocols.earth_mover_distance(::GenerativeModelProtocol, ::AbstractMatrix)
GenerativeModelProtocols.earth_mover_distance(::GenerativeModelProtocol)
GenerativeModelProtocols.energy_distance(::AbstractMatrix, ::AbstractMatrix)
GenerativeModelProtocols.energy_distance(::GenerativeModelProtocol, ::AbstractMatrix)
GenerativeModelProtocols.energy_distance(::GenerativeModelProtocol)
```

# Convenience Constructors
```@docs
GenerativeModelProtocols.GenerativeModelProtocol(::GenerativeModelProtocols.AbstractGenerativeModel, ::Matrix{<:AbstractFloat})
GenerativeModelProtocols.VariationalAutoencoder(::Tuple{Vararg{Lux.Chain}}, ::Tuple{Vararg{Lux.Chain}})
GenerativeModelProtocols.VariationalAutoencoder(::NamedTuple{Any, <:Tuple{Vararg{Lux.Chain}}}, ::NamedTuple{Any, <:Tuple{Vararg{Lux.Chain}}})
GenerativeModelProtocols.VariationalAutoencoder(::Int, ::Int, ::Int)
GenerativeModelProtocols.GaussianMixtureModel(::Int, ::Int)
GenerativeModelProtocols.DiffusionModel(::Int, ::Tuple{Vararg{AbstractFloat}})
GenerativeModelProtocols.DiffusionModel(::Int, ::F, ::F, ::GenerativeModelProtocols.TabularDenoiser{LayerNames}) where {LayerNames, F<:AbstractFloat}
GenerativeModelProtocols.DiffusionModel(::Int, ::Int, ::F, ::F) where {F<:AbstractFloat}
GenerativeModelProtocols.TabularDenoiser(::Int; ::Int, ::Int, ::Function, <:AbstractFloat)
GenerativeModelProtocols.GenerativeAdversarialNetwork(::Int, ::Int)
GenerativeModelProtocols.NormalizingFlow(::Int)
```

