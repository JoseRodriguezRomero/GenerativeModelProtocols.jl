# Sinkhorn Distance

The Sinkhorn distance between two random distributions $\mu$ and $\nu$ provides 
a regularized measure of the discrepancy between them by adding an entropic 
penalty to the classical optimal transport problem. In contrast to the empirical 
Earth Mover's Distance, which suffers heavily from the curse of dimensionality, 
or the energy distance, which collapses structural allocations into rigid 
average pairwise combinations, the Sinkhorn distance introduces a tuning 
parameter that smoothly interpolates between maximum-variance optimal transport 
and a fully decoupled geometry. To this end, the comparison scales efficiently 
via a matrix scaling regularization framework.

Let $X \sim \mu$ and $Y \sim \nu$ be random variables, and let $\Pi(\mu, \nu)$ 
denote the set of all valid joint distributions coupling $\mu$ and $\nu$. For a 
ground metric $d$ and a regularization parameter $\varepsilon > 0$, the relative 
entropy (Kullback-Leibler divergence) of a coupling $\pi$ with respect to the 
product marginal distribution $\mu \otimes \nu$ is introduced as a penalty term. 

Thus, the comparison between the potentially high-dimensional distributions 
$\mu$ and $\nu$ can be expressed as a regularized minimization over expected 
transport costs. The Sinkhorn distance is defined as
```math
    W_{1,\varepsilon}(\mu, \nu) \equiv \inf_{\pi \in \Pi(\mu, \nu)} 
        \mathbb{E}_{(X,Y) \sim \pi} [ d(X, Y) ] + \varepsilon \, 
        \mathrm{D}_{\mathrm{KL}}(\pi \,\Vert{}\, \mu \otimes \nu). 
```

The entropic penalty restricts the optimization space, smoothing the transport 
plan by spreading probability mass more broadly than unregularized transport. 
Consequently, as $\varepsilon \to 0$, $W_{1,\varepsilon}$ recovers the classical 
Earth Mover's Distance, whereas as $\varepsilon \to \infty$, it relaxes to a 
energy-like expression. Specifically,
```math
    \lim_{\varepsilon \to \infty} W_{1,\varepsilon}(\mu, \nu) = 
        \mathbb{E}_{X \sim \mu, Y \sim \nu} [d(X, Y)].
```

For distributions with finite first moments, the Sinkhorn distance is 
non-negative and is zero *if and only if* the two distributions are identical, 
assuming proper baseline shifting:
```math
    W_{1,\varepsilon}(\mu, \nu) = 0 \quad \Longleftrightarrow \quad \mu = \nu. 
```
Thus, it can be used as a sample-based measure of distributional discrepancy 
that stabilizes high-dimensional optimization routines.

## Empirical Estimation

For empirical samples
```math
    x_1, \dots, x_n \sim \mu \quad \text{and} \quad y_1, \dots, y_m \sim \nu, 
```
the continuous regularized transport optimization reduces to a discrete matrix 
scaling framework. This yields an empirical estimator 
$\widehat{W}_{1,\varepsilon}$ whose computational cost is dominated by 
matrix-vector multiplications via Sinkhorn’s scaling algorithm, making the 
method exceptionally fast and highly parallelizable on modern GPU hardware 
compared to standard linear programming solvers.

In other words, if $d$ is the Euclidean distance, then
```math
    \widehat{W}_{1,\varepsilon} \equiv \min_{\gamma \in \mathbb{R}^{n 
        \times m}_+} \sum_{i=1}^n \sum_{j=1}^m \gamma_{ij} \left \lVert x_i - 
        y_j \right \rVert + \varepsilon \sum_{i=1}^n \sum_{j=1}^m \gamma_{ij} 
        \log( n m \gamma_{ij}),
```
subject to the discrete marginal conservation constraints 
```math
    \sum_{j=1}^m \gamma_{ij} = \frac{1}{n} 
    \qquad \text{and} \qquad
    \sum_{i=1}^n \gamma_{ij} = \frac{1}{m}.
```

## Convergence Rate

For independent samples, the empirical Sinkhorn distance completely bypasses the 
dimensional bottleneck plaguing unregularized optimal transport. Because the 
entropic penalty restricts the effective degrees of freedom of the coupling to a 
smooth class of functions, the statistical variance of the estimator behaves 
like a parametric expectation rather than a space-filling quantization network. 

The smoothness induced by $\varepsilon$ ensures that fluctuations between 
localized sample clusters do not force the global coupling network to undergo 
macroscopic re-allocations. Nonzero correlations among transport assignments are 
globally smoothed out across the entire domain, preventing individual point 
dependencies from scaling exponentially with the dimension.

Consequently, for any fixed regularization parameter $\varepsilon > 0$, the 
empirical Sinkhorn estimator avoids the $\mathcal{O}(n^{-1/d})$ curse of 
dimensionality and achieves a dimension-free convergence rate. The empirical 
variance scales as:
```math
    \operatorname{Var}(\widehat{W}_{1,\varepsilon}) = \mathcal{O}\left( 
        \frac{1}{n} + \frac{1}{m} \right). 
```

For comparable sample sizes where $n \sim m$, this simplifies directly to:
```math
    \operatorname{Var}(\widehat{W}_{1,\varepsilon}) = \mathcal{O}(n^{-1}), 
```
which implies that the regularized estimator satisfies the standard parametric 
rate:
```math
    \widehat{W}_{1,\varepsilon} - W_{1,\varepsilon}(\mu, \nu) = 
        \mathcal{O}_p(n^{-1/2}). 
```

The standard $\mathcal{O}(n^{-1/2})$ rate applies universally across all 
dimensions for a fixed $\varepsilon$. The cases of $d=1$ and $d=2$ do not follow 
this uniform flat trend and exhibit distinct edge behaviors. If the 
dimension-free trend held perfectly, the low-dimensional profiles would look 
identical to the higher dimensions. However, when $d=1$ or $d=2$, the 
unregularized distance itself already matches or beats the parametric ceiling, 
meaning the addition of the entropic penalty mainly adjusts the underlying 
scaling constants rather than lifting a dimensional bottleneck. 

Importantly, this sample complexity properties demonstrate that Sinkhorn 
regularization acts as a bridge between optimal transport and maximum mean 
discrepancy or energy distance methods. While unregularized empirical $W_1$ 
suffers a severe rate degradation of $n^{-1/d}$ in dimensions $d > 2$, the 
Sinkhorn distance maintains a stable $n^{-1/2}$ exponent. This dimension-free 
convergence makes it well-suited for high-dimensional sample comparisons, though 
it introduces a structural trade-off where the exact geometry of the optimal 
transport plan is blurred by entropy.

