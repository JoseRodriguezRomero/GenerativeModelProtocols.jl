# Earth Mover's Distance

The Earth Mover's Distance (EMD), or Wasserstein-1 distance, between two random 
distributions $\mu$ and $\nu$ provides a measure of the discrepancy between them 
based on the minimal cost of transporting probability mass from one 
distribution to the other. In contrast to distribution distances that require 
explicit estimation of probability densities or independent pairwise average 
combinations, the EMD considers the geometry of the space by optimizing a 
transport plan between samples, making it intuitive for distributions where 
structural transformations matter.

Let $X \sim \mu$ and $Y \sim \nu$ be random variables with respective 
distributions. We define the set of all valid couplings (joint distributions) 
$\Pi(\mu, \nu)$ such that for any $\pi \in \Pi(\mu, \nu)$, the marginal 
distributions satisfy 
```math
    \int \pi(x, y) dy = \mu(x)
    \qquad \text{and} \qquad 
    \int \pi(x, y) dx = \nu(y)
```
where $d$ is a user-defined ground metric, *e.g.*, the Euclidean distance. Thus, 
the comparison between the potentially high-dimensional distributions $\mu$ and 
$\nu$ can be expressed in terms of an infimum over expected distances under 
these joint allocations. That is,
```math
    W_1(\mu, \nu) \equiv \inf_{\pi \in \Pi(\mu, \nu)} \mathbb{E}_{(X,Y) \sim 
        \pi} \left[ d(X, Y) \right],
```
or, equivalently, via the Kantorovich-Rubinstein duality theorem for 1-Lipschitz 
functions:
```math
    W_1(\mu, \nu) = \sup_{\lVert f \rVert_L \le 1} \left( \mathbb{E}_{X \sim 
    \mu}[f(X)] - \mathbb{E}_{Y \sim \nu}[f(Y)] \right). 
```

The first formulation measures the minimal expected transport effort between 
distributions, whereas the dual formulation characterizes the largest possible 
discrepancy exposed by any smooth, non-expanding function $f$. Consequently, 
$W_1(\mu, \nu)$ increases when probability mass must be moved across larger 
spatial regions to transform one distribution into the other.

For distributions with finite first moments, the EMD is non-negative and is 
zero *if and only if* the two distributions are identical:
```math
    W_1(\mu, \nu) = 0 \quad \Longleftrightarrow \quad \mu = \nu. 
```
Thus, it can be used as a sample-based measure of distributional discrepancy 
without explicitly estimating either probability density.

## Empirical Estimation

For empirical samples
```math
    x_1, \dots, x_n \sim \mu \quad \text{and} \quad y_1, \dots, y_m \sim \nu, 
```
the continuous distributions are approximated by their empirical counterparts 
```math
    \hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n \delta_{x_i}
    \qquad \text{and} \qquad
    \hat{\nu}_m = \frac{1}{m}\sum_{j=1}^m \delta_{y_j}.
``` 
This yields an empirical estimator $\widehat{W}_1$ whose computational cost is 
dominated by solving a linear programming problem or network simplex 
optimization over the transport cost matrix, making the method straightforward 
but computationally intensive for large multivariate reference datasets.

In other words, if $d$ is the Euclidean distance, then
```math
    \widehat{W}_1 \equiv \min_{\gamma \in \mathbb{R}^{n \times m}_+} 
    \sum_{i=1}^n \sum_{j=1}^m \gamma_{ij} \left \lVert x_i - y_j \right \rVert, 
```
subject to the discrete conservation constraints
```math
    \sum_{j=1}^m \gamma_{ij} = \frac{1}{n} 
    \qquad \text{and} \qquad 
    \sum_{i=1}^n \gamma_{ij} = \frac{1}{m}.
```

## Convergence Rate

For independent samples, the empirical EMD converges to the population distance, 
but its convergence rate depends heavily on the data's dimension $d$. To see 
this, consider first the isolated convergence of the empirical distribution to 
its true target:
```math
    \Delta_n = W_1(\hat{\mu}_n, \mu), 
```
which estimates the structural space-filling properties of the sample grid.

Although $\hat{\mu}_n$ contains $n$ discrete probability masses, their locations 
are tightly constrained by the global geometry of the underlying space. Two 
localized regions of the space share mass boundaries, and the task of transport 
optimization creates global sample dependencies across the entire coupling 
network. Nonzero correlations occur between the transport assignments of 
near-neighbor samples across the entire domain.

The number of cells required to partition a $d$-dimensional space grows 
exponentially, which means the expected distance from a random point to its 
nearest sample is tied to the dimensionality of the data. Consequently, if 
$d > 2$, the expected empirical quantization error scales as:
```math
    \mathbb{E}[W_1(\hat{\mu}_n, \mu)] = \mathcal{O}(n^{-1/d}).
```

For a fixed dimension $d$, this simplifies to a slower non-parametric rate:
```math
    \operatorname{Var}(W_1(\hat{\mu}_n, \mu)) = \mathcal{O}(n^{-2/d}),
```
which implies that the empirical distribution satisfies:
```math
    \hat{\mu}_n - \mu = \mathcal{O}_p(n^{-1/d}).
```

The same structural argument applies to the second target distribution. Consider 
the empirical convergence profile for $\nu$:
```math
    \Delta_m = W_1(\hat{\nu}_m, \nu). 
```

Letting the localized displacement error be bounded by space-filling bounds, its 
discrete variance behaves under the exact same geometric constraints.

If the two samples are drawn from high-dimensional spaces, the convergence 
slow-down manifests because discrete samples cannot easily capture continuous 
spatial densities without exponential sample counts. Concretely, there are grid 
partitions that scale as $n^{1/d}$ along each coordinate axes, leading to 
large gaps between the discretized distributions. Since the total global 
error is bounded by the triangle inequality of the metric space, the empirical 
transport error scales as:
```math
    \mathbb{E}[W_1(\hat{\nu}_m, \nu)] = \mathcal{O}(m^{-1/d}).
```

Taking the rate for comparable sample sizes yields the standard error scaling:
```math
    W_1(\hat{\nu}_m, \nu) = \mathcal{O}_p(m^{-1/d}).
```

The same result holds for the corresponding cross-transport error between $\mu$ 
and $\nu$. Since the total empirical EMD bound respects the absolute metric 
properties of optimal transport:
```math
    \lvert \widehat{W}_1 - W_1(\mu, \nu) \rvert \le W_1(\hat{\mu}_n, \mu) +
     W_1(\hat{\nu}_m, \nu), 
```
its statistical error therefore satisfies
```math
    \boxed{ \widehat{W}_1 - W_1(\mu, \nu) = 
        \mathcal{O}_p(n^{-1/d}) }
```
when the two sample sizes are of the same order. More generally, the error 
scales as
```math
    \mathcal{O}_p \left( n^{-1/d} + m^{-1/d} \right).
```

However, the standard $\mathcal{O}(n^{-1/d})$ rate only applies for dimensions 
$d > 2$. The cases of $d=1$ and $d=2$ do not follow this trend. Specifically, 
when $d=1$, the problem simplifies to aligning cumulative distribution 
functions, yielding the standard parametric rate:
```math
    \mathbb{E}[W_1(\hat{\mu}_n, \mu)] = \mathcal{O}(n^{-1/2}), 
```
and, when $d=2$, the spatial matching introduces a mild logarithmic bottleneck, 
resulting in the rate:
```math
    \mathbb{E}[W_1(\hat{\mu}_n, \mu)] = \mathcal{O}\left( 
        \sqrt{\frac{\log n}{n}} \right). 
```

Importantly, this convergence rate severely deteriorates for higher dimensional
samples, a phenomenon known as the curse of dimensionality. For empirical 
$W_1$, the convergence rate in dimensions $d>2$ is typically $n^{-1/d}$, 
reflecting the characteristic spacing between samples in $d$ dimensions. In 
contrast, energy-distance estimators bypass this bottleneck because they only 
require estimating global expectations of scalar pairwise distances. The 
dimension can affect the constants and variance of the energy distance, but its 
convergence exponent remains fixed at $n^{-1/2}$ under finite variance 
assumptions.

Thus, for high-dimensional distribution comparison, the EMD suffers from 
significantly less favorable sample convergence than the empirical energy 
distance. This is the trade-off for capturing explicit horizontal transport 
paths and matching geometry directly rather than collapsing the distribution 
profile into scalar pairwise expectations.

