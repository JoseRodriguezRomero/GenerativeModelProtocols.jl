# Generative Adversarial Networks

Generative Adversarial Networks (GANs) are popular generative models built on a 
distinct concept: they pit two primary components, a generator and a 
discriminator, against each other in a zero-sum game. The discriminator's job is 
to distinguish synthetic data from real data, while the generator focuses on 
producing synthetic data that the discriminator fails to differentiate from the 
real data.

In this module, the implementation uses Wasserstein distance to train the 
discriminator. Alongside this, an encoder trains simultaneously to map real data 
points back into the latent space. This encoder uses a cost function identical 
to the Evidence Lower Bound (ELBO) found in Variational Autoencoders (VAEs), 
making this implementation a hybrid WGAN-VAE.

Once training is complete, the generator operates as a standalone model. It 
takes stochastic latent variables from a simple distribution and transforms them 
into synthetic data points within the target data space.

```@raw html
<div style="text-align: center; margin: 1.5em 0;">
    <img src="gan_diagram.svg" alt="gan_diagram" 
        style="padding: 5px; 
               max-width: 750px; 
               width: 100%;
               height: auto; 
               border: 1px solid transparent;
               border-radius: 10px;
               background-image: linear-gradient(var(--sidebar-bg, #ffffff), var(--sidebar-bg, #ffffff)), linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 50%, #94a3b8 100%);
               background-origin: border-box;
               background-clip: content-box, border-box;
               box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
</div>
```

## Discriminator

## Generator

## Encoder

