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

k = [15, 65, 120]
models = [GenerativeModelProtocols.GaussianMixtureModel(2, ki) for ki in k]
protocols = [
    GenerativeModelProtocol(model, train_data;
        batchsize = 256,
        epochs = 4500,
        optimiser = Adam(; eta = 1.0E-3, beta = (0.95,0.999)),
        device = cpu_device()
    )
    for model in models
]

for i in eachindex(protocols)
    train!(protocols[i])
end

## Compare Synthetic Data
synthetic_data_1 = protocols[1](num_samples)
synthetic_data_2 = protocols[2](num_samples)
synthetic_data_3 = protocols[3](num_samples)

p1 = scatter(x_train, y_train, title="Training data", label=false,frame=:box)
p2 = scatter(synthetic_data_1[:,1], synthetic_data_1[:,2], title="Synthetic data (k = $(k[1]))", label=false,frame=:box)
p3 = scatter(synthetic_data_2[:,1], synthetic_data_2[:,2], title="Synthetic data (k = $(k[2]))", label=false,frame=:box)
p4 = scatter(synthetic_data_3[:,1], synthetic_data_3[:,2], title="Synthetic data (k = $(k[3]))", label=false,frame=:box)
p = plot(p1,p2,p3,p4; layout=(2,2), size=(900, 600))
savefig(p, "intro_example.svg")

## Compare Distributions
function plot_model_density(protocol)
    data = protocol(100000)
    
    fig = histogram2d(data[:,1], data[:,2],
        bins=(200,200),
        show_empty_bins=true,
        title="GMM PDF (k = $(protocol.model.k))\n2D Histogram",
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
p2 = plot_model_density(protocols[1])
p3 = plot_model_density(protocols[2])
p4 = plot_model_density(protocols[3])
p = plot(p1, p2, p3, p4; layout = (2,2), size = (900, 800))
savefig(p, "intro_example_density_comp.svg")

