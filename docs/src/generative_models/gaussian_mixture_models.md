# Gaussian Mixture Models

Gaussian Mixture Models (GMMs) are among the simplest generative models capable 
of learning an unknown probability density distribution from a data collection. 
Generally, a GMM consists of two primary operational steps: component assignment 
and conditional emission. The goal of the component assignment step is to map a 
recorded observation to a lower-dimensional discrete latent variable 
representing cluster membership. This latent variable follows a categorical 
probability distribution based on mixture weights that is easy to sample from. 
Conversely, the conditional emission step transforms these sampled latent 
components back into the original data space by generating values from the 
selected component's specific Gaussian distribution.

Once a GMM is trained, it can be used as a generative model. Because the latent 
space consists of a discrete set of sub-populations, generating new random 
observations requires first sampling a specific component from the model's 
categorical distribution. Each of these latent components is defined by its own 
mean vector and covariance matrix, allowing the model to capture complex, 
multi-modal data structures.

```@raw html
<div style="text-align: center; margin: 1.5em 0;">
    <img src="gmm_diagram.svg" alt="gmm_diagram" 
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

