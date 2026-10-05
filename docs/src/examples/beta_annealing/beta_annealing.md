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
training_data      = 2×5000 Matrix{AbstractFloat}
mean_training_data = Tuple{Float64, Float64}
var_training_data  = Tuple{Float64, Float64}
model              = GenerativeModelProtocols.VariationalAutoencoder
```
Since the `protocol.model` is a 
`GenerativeModelProtocols.VariationalAutoencoder`, we can, for example, 
train our VAE model with $\beta = 0.1$ by invoking
```julia-repl
julia> train!(protocol; β = 0.1, batchsize = 256, epochs = 50);
Training VAE... (β = 0.1)
Epoch        1 | Avg. ELBO:  -9.97973323e-01 
Epoch        5 | Avg. ELBO:  -9.65876400e-01 
Epoch       10 | Avg. ELBO:  -8.33865821e-01 
Epoch       15 | Avg. ELBO:  -7.56704390e-01 
Epoch       20 | Avg. ELBO:  -7.27888465e-01 
Epoch       25 | Avg. ELBO:  -7.23917782e-01 
Epoch       30 | Avg. ELBO:  -6.97996140e-01 
Epoch       35 | Avg. ELBO:  -6.12449586e-01 
Epoch       40 | Avg. ELBO:  -3.84669602e-01 
Epoch       45 | Avg. ELBO:  -3.42154860e-01 
Epoch       50 | Avg. ELBO:  -3.32174629e-01 
Training complete!
Training took 5 seconds.
```
Likewise, we can continue training our VAE model with a different $\beta$, in
this example $\beta = 0.5$, for another 80 epochs by simply invoking
```julia-repl
julia> train!(protocol; β = 0.5, batchsize = 256, epochs = 80);
Training VAE... (β = 0.5)
Epoch        1 | Avg. ELBO:  -1.11497259e+00 
Epoch        5 | Avg. ELBO:  -8.73123646e-01 
Epoch       10 | Avg. ELBO:  -8.45595360e-01 
Epoch       15 | Avg. ELBO:  -8.35553467e-01 
Epoch       20 | Avg. ELBO:  -8.19539189e-01 
Epoch       25 | Avg. ELBO:  -8.32773864e-01 
Epoch       30 | Avg. ELBO:  -8.15521598e-01 
Epoch       35 | Avg. ELBO:  -8.19420218e-01 
Epoch       40 | Avg. ELBO:  -8.29790711e-01 
Epoch       45 | Avg. ELBO:  -8.17085862e-01 
Epoch       50 | Avg. ELBO:  -8.24123561e-01 
Epoch       55 | Avg. ELBO:  -8.17595303e-01 
Epoch       60 | Avg. ELBO:  -8.21818233e-01 
Epoch       65 | Avg. ELBO:  -8.20839107e-01 
Epoch       70 | Avg. ELBO:  -8.20317924e-01 
Epoch       75 | Avg. ELBO:  -8.22621346e-01 
Epoch       80 | Avg. ELBO:  -8.23238373e-01 
Training complete!
Training took 4 seconds.
```

