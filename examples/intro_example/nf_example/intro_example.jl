using GenerativeModelProtocols
using Lux, Plots, StatsBase
using LaTeXStrings

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

model = GenerativeModelProtocols.NormalizingFlow(2)
protocol = GenerativeModelProtocol(model,train_data;
    batchsize   = 256,
    epochs      = 1500,
    optimiser   = Adam(; eta = 1.0E-3, beta = (0.95,0.999)),
    device      = cpu_device()
)
train!(protocol)

