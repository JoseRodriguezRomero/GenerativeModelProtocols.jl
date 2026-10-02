using GenerativeModelProtocols
using DifferentialEquations
using FileIO, HDF5
using Optimisers
using Reactant
using Lux

include("make_data.jl")

num_samples = 5000
x_train, y_train = make_data(num_samples)
train_data = collect(transpose(hcat(x_train,y_train)))

model = GenerativeModelProtocols.NormalizingFlow(2)
protocol = GenerativeModelProtocol(model, train_data;
    batchsize = 256,
    epochs = 3500,
    optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999)),
    device = reactant_device()
)
nf_train_log = train!(protocol)
save("trained_models/nf_model.h5", protocol)

model = GenerativeModelProtocols.DiffusionModel(2, 250)
protocol = GenerativeModelProtocol(model, train_data;
    batchsize = 256,
    epochs = 3500,
    optimiser = Adam(; eta = 1.0E-3, beta = (0.95,0.999)),
    device = reactant_device()
)
dm_train_log = train!(protocol)
save("trained_models/dm_model.h5", protocol)

model = GenerativeModelProtocols.GaussianMixtureModel(2, 65)
protocol = GenerativeModelProtocol(model, train_data;
    batchsize = 256,
    epochs = 3500,
    optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999)),
    device = reactant_device()
)
gmm_train_log = train!(protocol)
save("trained_models/gmm_model.h5", protocol)

model = GenerativeModelProtocols.VariationalAutoencoder(2, 2)
protocol = GenerativeModelProtocol(model, train_data;
    batchsize = 256,
    epochs = 3500,
    optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999)),
    device = reactant_device()
)
vae_train_log = train!(protocol; β = 0.1)
save("trained_models/vae_model.h5", protocol)

critic_optimiser = Adam(; eta = 1.0E-4, beta = (0.0,0.9))
vae_optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999))

model = GenerativeModelProtocols.GenerativeAdversarialNetwork(2, 2)
protocol = GenerativeModelProtocol(model, train_data;
    batchsize = 256,
    epochs = 3500,
    optimiser = (critic_optimiser, vae_optimiser),
    device = reactant_device()
)
gan_train_log = train!(protocol; β = 0.1, γ_vae = 1.0, γ_wgan = 0.1,
    grad_penalty = true, λ = 10.0, a = 1.0,
    weight_clipping = false, clip_value = 0.05
)
save("trained_models/gan_model.h5", protocol)

open("timings.txt", "w") do io
    println(io, "nf_model $(nf_train_log.elapsed)")
    println(io, "dm_model $(dm_train_log.elapsed)")
    println(io, "gmm_model $(gmm_train_log.elapsed)")
    println(io, "vae_model $(vae_train_log.elapsed)")
    println(io, "gan_model $(gan_train_log.elapsed)")
end

