# Diffusion Models

To demonstrate how to use this module, we will train a Diffusion Model
(DM) [Sohl2015](@cite) to generate synthetic data that mirrors a 
two-dimensional target distribution. The DM must learn to mimic this 
distribution using only the provided training samples, without direct access to 
the true underlying distribution or the ability to sample additional data.

For this example we will need to load the following modules
```julia
using Flux
using GenerativeModelProtocols
```
let us consider the following function to generate our random two-dimensional 
data
```julia
function make_data(num_samples)
    t = (2.0*π) .* rand(Float64,num_samples)

    x_noise = 0.05 .* randn(Float64,num_samples)
    y_noise = 0.05 .* randn(Float64,num_samples)

    x = cos.(1.0.*t) .+ x_noise
    y = sin.(2.0.*t) .+ y_noise
    
    return x, y
end

num_samples = 5000
x_train, y_train = make_data(num_samples)
train_data = collect(transpose(hcat(x_train,y_train)))
```
we then define the DM that is to be trained on this data, whose latent 
variables are also two-dimensional as
```julia
model = GenerativeModelProtocols.DiffusionModel(2, 250)
```
thus, our trainable generative model protocol is defined, and trained, as
```julia
protocol = GenerativeModelProtocol(model, train_data;
    batchsize   = 256,
    epochs      = 1500,
    optimiser   = Adam(; eta = 1.0E-3, beta = (0.95,0.999)),
    device      = cpu_device()
)
train!(protocol)
```
finally, we can generate synthetic data by simply invoking our trained protocol
```julia
synthetic_data = protocol(num_samples)
```

```@raw html
<div style="text-align: center; margin: 1.5em 0;">
    <img src="intro_example.svg" alt="intro_example" 
        style="padding: 5px; 
               max-width: 750px; 
               width: 100%;
               height: auto; 
               border: 1px solid transparent;
               border-radius: 10px;
               background-image: 
                linear-gradient(var(--sidebar-bg, #ffffff), var(--sidebar-bg, #ffffff)), 
                linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 50%, #94a3b8 100%);
               background-origin: border-box;
               background-clip: content-box, border-box;
               box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
</div>
```

## Displaying models and protocols

As part of this module, `Base.display` is overloaded with most relevant 
structures that users can need. We can easily read the configuration of 
`protocol` from the REPL by inputting
```julia-repl
julia> protocol
GenerativeModelProtocol{GenerativeModelProtocols.DiffusionModel}:
training_data = 2×5000 Matrix{Float64}
epochs        = 1500
batchsize     = 256
shuffle       = true
optimiser     = Adam(eta=0.001, beta=(0.95, 0.999), epsilon=1.0e-8)
device        = CPUDevice
model         = GenerativeModelProtocols.DiffusionModel

```
likewise, we can also read the configuration of `model` from the REPL by
inputting
```julia-repl
julia> model
GenerativeModelProtocols.DiffusionModel:
α              = NTuple{250, Float64}
ᾱ              = NTuple{250, Float64}
β              = NTuple{250, Float64}
denoiser_model = GenerativeModelProtocols.TabularDenoiser
```

## Model validation

So far, we have only validated our model by verifying that its synthetic data 
resembles the training data. However, we can perform at least two additional 
tests to assess whether the model is functioning as intended:
* __Test Data Reconstruction__: Encode and decode a separate, unseen test 
  dataset to see if the reconstructed data closely resembles the original.
* __Latent Space Validation__: Verify that all latent variables in our trained 
  DM act as statistically independent random variables following a standard 
  normal distribution (zero mean and unit variance).

For our given example, we can easily test the ability of our trained model to 
reconstruct some inputted data by simply calling 
`GenerativeModelProtocols.encode` and `GenerativeModelProtocols.decode` as 
follows
```julia
x_test, y_test = make_data(800)
test_data = collect(transpose(hcat(x_test,y_test)))
z_test_data = GenerativeModelProtocols.encode(model,test_data)
recon_test_data = GenerativeModelProtocols.decode(model,z_test_data)
```
```@raw html
<div style="text-align: center; margin: 1.5em 0;">
    <img src="intro_example_recon.svg" alt="intro_example_recon" 
        style="padding: 5px; 
               max-width: 750px; 
               width: 100%;
               height: auto; 
               border: 1px solid transparent;
               border-radius: 10px;
               background-image: 
                linear-gradient(var(--sidebar-bg, #ffffff), var(--sidebar-bg, #ffffff)), 
                linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 50%, #94a3b8 100%);
               background-origin: border-box;
               background-clip: content-box, border-box;
               box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
</div>
```

To test if the latent variables of our trained model follow a normal 
distribution with zero mean and unit variance we solely need to use the 
DM's noise model, however, unlike our previous reconstruction example it is 
preferable to use a larger test dataset
```julia
x_test, y_test = make_data(10000)
test_data = collect(transpose(hcat(x_test,y_test)))
z_test_data = GenerativeModelProtocols.encode(model,test_data)
```
```@raw html
<div style="text-align: center; margin: 1.5em 0;">
    <img src="intro_example_latents.svg" alt="intro_example_latents" 
        style="padding: 5px; 
               max-width: 750px; 
               width: 100%;
               height: auto; 
               border: 1px solid transparent;
               border-radius: 10px;
               background-image: 
                linear-gradient(var(--sidebar-bg, #ffffff), var(--sidebar-bg, #ffffff)), 
                linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 50%, #94a3b8 100%);
               background-origin: border-box;
               background-clip: content-box, border-box;
               box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
</div>
```

To evaluate whether the latent variables of our trained DM behave as 
independent random variables by calculating the Pearson correlation coefficient 
or, more robustly, the Spearman rank correlation coefficient.
```julia
using StatsBase

pearson_corr = cor(z_test_data[1,:],z_test_data[2,:])
spearman_corr = corspearman(z_test_data[1,:],z_test_data[2,:])
```
```@raw html
<div style="text-align: center; margin: 1.5em 0;">
    <img src="intro_example_latents_corr.svg" alt="intro_example_latents_corr" 
        style="padding: 5px; 
               max-width: 750px; 
               width: 100%;
               height: auto; 
               border: 1px solid transparent;
               border-radius: 10px;
               background-image: 
                linear-gradient(var(--sidebar-bg, #ffffff), var(--sidebar-bg, #ffffff)), 
                linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 50%, #94a3b8 100%);
               background-origin: border-box;
               background-clip: content-box, border-box;
               box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
</div>
```
the resulting correlations, visible in the illustration above, are close enough 
to zero to confirm that the latent variables are effectively independent.

## Source code

The scripts used to create and train the DM, and to produce all the plots shown 
in this page are included in the `examples` folder of this module.
```bash
cd /path/to/GenerativeModelProtocols.jl/examples/intro_example/dm_example
julia intro_example.jl
```

## References
```@bibliography
Pages = [@__FILE__]
```
