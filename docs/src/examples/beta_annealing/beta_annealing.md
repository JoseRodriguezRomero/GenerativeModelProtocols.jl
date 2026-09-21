# $\beta$ Annealing

Recall that the objective function used for training is an Evidence Lower
Bound (ELBO) of the log-likelihood of the observed training data, obtained using 
Jensen's inequality. The objective function used for training a 
Variational Autoencoder (VAE) consists of two components: a reconstruction 
loss term and a Kullback-Leibler (KL) divergence term. In other words:
```math
    \text{ELBO}_{\phi, \theta} = \mathbb{E}_{q_\phi \left( z \left| x \right. 
        \right)} \left[ \log \left( p_\theta \left( x | z \right) \right) 
        \right] - D_\text{KL} \left( q_\phi \left( z \left| x \right. \right) 
        \ \lVert \ p(z) \right).
```

A common issue encountered when training VAEs with this ELBO is posterior 
collapse, where the decoder ignores the latent variables and the encoder ignores 
the input, leading to $q_\phi \left( z \left| x \right. \right) \approx p(z)$. 
A widely adopted solution is to introduce a hyperparameter $\beta$ that balances 
the reconstruction loss against the KL divergence term, modifying the objective 
to:
```math
    \text{ELBO}_{\phi, \theta} = \mathbb{E}_{q_\phi \left( z \left| x \right. 
        \right)} \left[ \log \left( p_\theta \left( x | z \right) \right) 
        \right] - \beta \ D_\text{KL} \left( q_\phi \left( z \left| x \right. 
        \right) \ \lVert \ p(z) \right).
```

Moreover, another strategy often used when training VAEs is to use an annealing 
schedule for the values of $\beta$ [Chunyuan2019](@cite), where, for example, 
one could begin with $\beta = 0$ and gradually increase it to a target value, 
such as $\beta = 0.1$.

This module offers both the option to train a VAE with either a fixed value of 
$\beta$, or establish some user-defined schedule. Suppose then that we have some
VAE model 
```julia-repl
julia> protocol
GenerativeModelProtocol:
training_data = 2×5000 Matrix{Float32}
epochs        = 40
batchsize     = 256
shuffle       = true
optimiser     = Adam(eta=0.0001, beta=(0.95f0, 0.999f0), epsilon=1.0e-8)
device        = CPUDevice
model         = GenerativeModelProtocols.VariationalAutoencoder
```
Since the `protocol.model` is a 
`GenerativeModelProtocols.VariationalAutoencoder`, we can, for example, 
train our VAE model with $\beta = 0.1$ by invoking
```julia-repl
julia> train!(protocol; β = 0.1);
Training VAE... (β = 0.1)
Epoch        1 | Avg. ELBO:   1.01004767e+00 
Epoch        5 | Avg. ELBO:   1.00131965e+00 
Epoch       10 | Avg. ELBO:   9.75843072e-01 
Epoch       15 | Avg. ELBO:   7.83954382e-01 
Epoch       20 | Avg. ELBO:   7.06776321e-01 
Epoch       25 | Avg. ELBO:   4.49170738e-01 
Epoch       30 | Avg. ELBO:   3.74058574e-01 
Epoch       35 | Avg. ELBO:   3.50416392e-01 
Epoch       40 | Avg. ELBO:   3.42971653e-01 
Training complete!
```
Likewise, we can train our VAE model with a $\beta$ annealing schedule in which 
$\beta = 0$ for the first 40 epochs, since `protocol.epochs` is in this example
equal to 40, and $\beta = 0.1$ for the last 40 epochs by invoking
```julia-repl
julia> train!(protocol; β = [0.0, 0.1]);
Training VAE... (β = 0.0)
Epoch        1 | Avg. ELBO:   1.09802745e-01 
Epoch        5 | Avg. ELBO:   4.43852916e-02 
Epoch       10 | Avg. ELBO:   2.08726693e-02 
Epoch       15 | Avg. ELBO:   1.05176121e-02 
Epoch       20 | Avg. ELBO:   6.07344974e-03 
Epoch       25 | Avg. ELBO:   3.92770581e-03 
Epoch       30 | Avg. ELBO:   2.74951989e-03 
Epoch       35 | Avg. ELBO:   2.13511358e-03 
Epoch       40 | Avg. ELBO:   1.71649177e-03 
Training complete!
Training VAE... (β = 0.1)
Epoch        1 | Avg. ELBO:   1.04711866e+00 
Epoch        5 | Avg. ELBO:   5.83685338e-01 
Epoch       10 | Avg. ELBO:   3.77804756e-01 
Epoch       15 | Avg. ELBO:   3.37563813e-01 
Epoch       20 | Avg. ELBO:   3.28588426e-01 
Epoch       25 | Avg. ELBO:   3.21542203e-01 
Epoch       30 | Avg. ELBO:   3.17488939e-01 
Epoch       35 | Avg. ELBO:   3.18278909e-01 
Epoch       40 | Avg. ELBO:   3.11614692e-01 
Training complete!
```

