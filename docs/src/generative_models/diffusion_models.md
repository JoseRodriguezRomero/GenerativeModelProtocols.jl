# Diffusion Models

Diffusion Models (DMs) are highly popular generative models that, like 
Variational Autoencoders (VAEs), feature a conceptually straightforward 
architecture, despite operating on completely different principles. While VAEs 
are parameterized by an encoder and a decoder neural network, DMs are 
parameterized by a denoiser model (typically a neural network) and a noise 
schedule structured as a Markov chain. Furthermore, both frameworks feature a 
latent space; however, unlike a VAE's compressed latent bottleneck, the latent 
space of a DM retains the exact dimensions of the input data.

Once a DM is trained, its denoiser model and reverse noise schedule can be used 
as a generative model. This generation process is possible because the forward 
process forces the final latent variables to become statistically independent, 
allowing us to simply seed the network with pure random noise to synthesize 
completely new data.
