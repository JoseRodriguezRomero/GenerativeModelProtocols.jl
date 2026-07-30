using GenerativeModelProtocols
using Flux, Plots, StatsBase
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

model = GenerativeModelProtocols.DiffusionModel(2,150, 1.0E-4, 1.0E-3)
protocol = GenerativeModelProtocol(model,train_data;
    batchsize   = 256,
    epochs      = 500,
    optimiser   = Adam(; eta = 1.0E-3, beta = (0.95,0.999)),
    device      = cpu_device()
)
train!(protocol)

synthetic_data = protocol(num_samples)

p11 = scatter(x_train, y_train, title="Training data", label=false,frame=:box)
p22 = scatter(synthetic_data[1,:], synthetic_data[2,:], title="Synthetic data", label=false,frame=:box)
pp = plot(p11,p22; layout=(2,1))

## Compare Distributions
function plot_model_density(protocol)
    data = protocol(50000)
    
    fig = histogram2d(data[1,:], data[2,:],
        bins=(200,200),
        show_empty_bins=true,
        title="DM PDF (T = $(protocol.model.T))\n2D Histogram",
        xlims=(-1.1,1.1),
        ylims=(-1.1,1.1),
        fillcolor=:viridis,
        aspect_ratio=:equal,
        colorbar=true,
        normalize=:pdf
    )
    
    return fig
end

function plot_real_density()
    x_test, y_test = make_data(50000)
    
    fig = histogram2d(x_test, y_test,
        bins=(200,200),
        show_empty_bins=true,
        title="Exact Data PDF\n2D Histogram",
        xlims=(-1.1,1.1),
        ylims=(-1.1,1.1),
        fillcolor=:viridis,
        aspect_ratio=:equal,
        colorbar=true,
        normalize=:pdf
    )
    
    return fig
end

p1 = plot_real_density()
p2 = plot_model_density(protocol)

p = plot(p1,p2; layout=(1,2),size = (900, 450))

