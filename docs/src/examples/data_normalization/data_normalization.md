# Data Normalization

Models learn by reducing errors in their input features. When one feature has 
values thousands of times larger than another, its errors can dominate training 
and hide patterns in the smaller-scale feature. Normalization centers a feature 
by subtracting its mean, then scales it by its standard deviation. This puts 
selected features on comparable scales so the model can learn from each more 
evenly. The protocol reverses this transformation when returning generated data.

The examples use one feature per row and one observation per column. Full 
normalization scales every feature, while hybrid normalization scales only 
features whose variance meets `hybrid_normalization_threshold`. Features below 
that threshold are left in their original scale, which is useful for constant 
or nearly constant features. Manual normalization lets users choose which 
features to scale.

```julia
using GenerativeModelProtocols

model = GenerativeModelProtocols.NormalizingFlow(2)
x = vcat(1.0E3 .* randn(Float32, 1, 100), 1.0E-4 .* randn(Float32, 1, 100))
```

## No normalization

Keep every feature in its original scale:

```julia
protocol = GenerativeModelProtocol(model, x;
    normalize_data = GenerativeModelProtocols.no_normalization)
```

## Full normalization

Standardize every feature to zero mean and unit variance. This divides each feature by its standard deviation, so constant or nearly constant features can produce unstable values:

```julia
full_x = vcat(1.0E3 .* randn(Float32, 1, 100), 1.0E-1 .* randn(Float32, 1, 100))
protocol = GenerativeModelProtocol(model, full_x;
    normalize_data = GenerativeModelProtocols.full_normalization)
```

## Hybrid normalization

Normalize features whose variance is at least `hybrid_normalization_threshold` 
(default is `1.0E-6`), leaving lower-variance features unchanged. Here, the 
first feature is normalized and the second stays as-is. This is the default 
mode.

```julia
protocol = GenerativeModelProtocol(model, x;
    normalize_data = GenerativeModelProtocols.hybrid_normalization)
```

## Manual normalization

Pass one Boolean per feature: `true` normalizes it, `false` leaves it unchanged.

```julia
protocol = GenerativeModelProtocol(model, x;
    normalize_data = (true, false))
```
