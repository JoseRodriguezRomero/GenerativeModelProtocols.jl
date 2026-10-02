# Energy Distance

The energy distance between two random distributions $\mu$ and $\nu$ provides a
measure of the discrepancy between them based solely on pairwise distances
between samples. In contrast to distribution distances that require explicit
estimation of probability densities, the energy distance can be evaluated
directly from samples, making it particularly convenient for comparing
high-dimensional distributions. To this end, the comparison is reduced to using 
one-dimensional random distributions of pairwise distances.

Let $X,X' \sim \mu$ be independent and identically distributed random variables,
and let $Y,Y' \sim \nu$ be independent and identically distributed random
variables. We define
```math
    \phi_{XX} \equiv \left\{ \left. d \left( X, X' \right) \ \right| \ 
        X,X' \sim \mu \right\} \\[0.2cm]

    \phi_{YY} \equiv \left\{ \left. d \left( Y, Y' \right) \ \right| \ Y,Y' \sim 
        \nu \right\},
```
and
```math
    \phi_{XY} \equiv \left\{ \left. d \left( X, Y \right) \ \right|\ X \sim 
        \mu,\ Y \sim \nu \right\},
```
where $d$ is a user-defined distance function, *e.g.*, the Euclidean distance.

Thus, the comparison between the potentially high-dimensional distributions
$\mu$ and $\nu$ can be expressed in terms of expectations over these
one-dimensional distributions of pairwise distances. The energy distance is then 
defined as
```math
    D_{\mu,\nu}
    \equiv
    2\,\mathbb{E}_{X \sim \mu, Y \sim \nu} \left[ d \left( X, Y \right) \right] -
    \mathbb{E}_{X, X' \sim \mu} \left[ d \left( X, X' \right) \right] -
    \mathbb{E}_{Y, Y' \sim \nu} \left[ d \left( Y, Y' \right) \right], 
```
or, equivalently,
```math
    D_{\mu,\nu}
    =
    2\,\mathbb{E}_{d_{XY}\sim\phi_{XY}}[d_{XY}]
    -
    \mathbb{E}_{d_{XX}\sim\phi_{XX}}[d_{XX}]
    -
    \mathbb{E}_{d_{YY}\sim\phi_{YY}}[d_{YY}].
```
The first term measures the expected distance between samples drawn from 
different distributions, whereas the second and third terms characterize the 
corresponding within-distribution distances. Consequently, $D_{\mu,\nu}$ 
increases when samples from $\mu$ and $\nu$ are, on average, farther apart than 
samples drawn within each distribution.

For distributions with finite first moments, the energy distance is 
non-negative and is zero *if and only if* the two distributions are identical:
```math
    D_{\mu,\nu}=0
    \quad \Longleftrightarrow \quad
    \mu=\nu.
```
Thus, it can be used as a sample-based measure of distributional discrepancy 
without explicitly estimating either probability density.

## Empirical Estimation

For empirical samples
```math
    x_1, \dots, x_n \sim \mu \quad \text{and} \quad y_1, \dots, y_m \sim \nu,
```
the expectations can be estimated from the corresponding pairwise distances.
This yields an estimator $\widetilde{D}_{\mu, \nu}$ whose computational cost is
dominated by the evaluation of the pairwise distance matrices, making the method
straightforward to apply to multivariate generated and reference data. 

In other words, if $d$ is the Euclidean distance, then
```math
    \widetilde{D}_{\mu, \nu} \equiv \frac{1}{n m} \sum_{i = 1}^n \sum_{j = 1}^m 
        \left \lVert x_i - y_j \right \rVert -
    \frac{2}{n (n - 1)} \sum^n_{i = 1} \sum_{j = i + 1}^n
        \left \lVert x_i - x_j \right \rVert -
    \frac{2}{m (m - 1)} \sum^m_{i = 1} \sum_{j = i + 1}^m
        \left \lVert y_i - y_j \right \rVert.
```

## Convergence Rate

For independent samples with finite second moments, the empirical energy 
distance converges to the population energy distance at the standard Monte Carlo 
rate. To see this, consider first the cross-distribution term
```math
    \widehat{A} = \frac{1}{nm} \sum_{i=1}^{n} \sum_{j=1}^{m} d(X_i, Y_j),
```
which estimates
```math
    A = \mathbb{E}[d(X, Y)].
```

Although $\widehat{A}$ contains $nm$ pairwise distances, these terms are not 
independent. Two terms $d(X_i, Y_j)$ and $d(X_k, Y_l)$ are independent when $i 
\neq k$ and $j \neq l$, as they involve disjoint random variables. Nonzero 
covariance occurs only when the two terms share an $X$ sample ($i = k$) or a 
$Y$ sample ($j = l$). 

The number of such dependent pairs is of order $\mathcal{O}(nm^2 + n^2m)$, 
whereas the total number of terms in the double summation for the variance is 
$n^2m^2$. Consequently, the variance scales as:
```math
    \operatorname{Var}(\widehat{A}) = \mathcal{O}\left( \frac{1}{n} + 
    \frac{1}{m} \right). 
```

For comparable sample sizes where $n \sim m$, this simplifies to:
```math
    \operatorname{Var}(\widehat{A}) = \mathcal{O}(n^{-1}), 
```
which implies that the estimator satisfies:
```math
    \widehat{A} - A = \mathcal{O}_p(n^{-1/2}). 
```

The same argument applies to the within-distribution term. Consider the 
estimator:
```math
    \widehat{B} = \frac{2}{n(n-1)} \sum_\){i<j} d(X_i, X_j).
```

Letting $Z_{ij} = d(X_i, X_j)$, its variance can be written as:
```math
    \operatorname{Var}(\widehat{B}) = \frac{4}{n^2(n-1)^2} \sum_{i<j} \sum_{k<l} 
    \operatorname{Cov} (Z_{ij}, Z_{kl}).
```

If the two pairs $(i, j)$ and $(k, l)$ share no common indices (i.e., they 
consist of four distinct samples), then $Z_{ij}$ and $Z_{kl}$ are independent, 
and their covariance vanishes. A nonzero covariance can only occur when the two 
pairs overlap by sharing at least one index.

Concretely, there are exactly $\binom{n}{2}$ pairwise distances, and each pair 
shares an index with $2(n-2)$ other pairs, leading to $\mathcal{O}(n^3)$ 
dependent covariance terms. Since the front normalization factor is proportional 
to $n^{-4}$, the total variance scales as:
```math
    \operatorname{Var}(\widehat{B}) = \mathcal{O}\left(\frac{1}{n}\right). 
```

Taking the square root yields the standard deviation:
```math
    \operatorname{SD}(\widehat{B}) = \mathcal{O}\left(\frac{1}{\sqrt{n}}\right). 
```

The same result holds for the corresponding within-distribution term for $\nu$. 
The corresponding result holds for the within-$\nu$ term. Since the empirical 
energy distance is a linear combination of these three estimates,
```math
    \widetilde{D}_{\mu,\nu} = 2 \widehat{A} - \widehat{B} - \widehat{C},
```
its statistical error therefore satisfies
```math
    \boxed{ \widetilde{D}_{\mu, \nu} - D_{\mu, \nu} = O_p(n^{-1/2}) }
```
when the two sample sizes are of the same order. More generally, the error 
scales as
```math
    O_p \left( n^{-1/2} + m^{-1/2} \right).
```

Importantly, this convergence rate does not deteriorate with the ambient 
dimension in the same way as the empirical Wasserstein distance. For empirical 
$W_1$, the convergence rate in dimensions $d>2$ is typically $n^{-1/d}$, 
reflecting the characteristic spacing between samples in $d$ dimensions. In 
contrast, the energy-distance estimator only requires estimating expectations of 
pairwise distances. The dimension can affect the constants and variance of the 
estimator, but not the $n^{-1/2}$ convergence exponent under the 
finite-second-moment assumption used above.

Thus, for high-dimensional distribution comparison, the energy distance can 
provide substantially more favorable sample convergence than empirical 
Wasserstein distance. This comes at the cost of measuring distributional 
discrepancy through pairwise distances rather than through an optimal transport 
plan.

