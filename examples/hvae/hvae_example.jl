using GenerativeModelProtocols
using Lux, Optimisers, Plots
using LaTeXStrings, Plots.PlotMeasures

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

L = [1, 2]

models = [GenerativeModelProtocols.VariationalAutoencoder(2, 2, l) for l in L]
protocols = [GenerativeModelProtocol(model, train_data;
    batchsize = 256,
    epochs = 3500,
    optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999)),
    device = cpu_device()
) for model in models]

for i in eachindex(protocols)
    train!(protocols[i]; β = 0.1)
end

## Compare density
synthetic_data = [protocol(num_samples) for protocol in protocols]

p1 = scatter(x_train, y_train, title="Training data", label=false,frame=:box)
p2 = scatter(synthetic_data[1][1,:], synthetic_data[1][2,:], title="Synthetic data", label=false,frame=:box)
p3 = scatter(synthetic_data[2][1,:], synthetic_data[2][2,:], title="Synthetic data", label=false,frame=:box)

function plot_model_density(protocol, L)
    data = protocol(50000)
    
    fig = histogram2d(data[1,:], data[2,:],
        bins=(200,200),
        show_empty_bins=true,
        title="HVAE (L = $L) PDF\n2D Histogram",
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

p11 = plot_real_density()
p22 = plot_model_density(protocols[1], L[1])
p33 = plot_model_density(protocols[2], L[2])

p = plot(p1,p11,p2,p22,p3,p33; layout=(3,2), size = (900, 1200), link = :x)
savefig(p, "hvae_example.svg")

## Compare latents
x_test, y_test = make_data(10000)
test_data = collect(transpose(hcat(x_test,y_test)))
z_test_data_1 = GenerativeModelProtocols.encode(protocols[1],test_data)
z_test_data_2 = GenerativeModelProtocols.encode(protocols[2],test_data)

function reference_gaussian(x, μ, σ²)
    return exp(-((x - μ)^2)/(2.0 * σ²))/sqrt(2.0*π*σ²)
end

function plot_reference_gaussian!()
    x = collect(-10.0:0.01:10)
    return plot!(x, reference_gaussian.(x,0,1), style=:dash, label=L"\mathcal{N} (0,1)", linewidth=5)
end

p1 = histogram(z_test_data_1[1,:],normalize=:pdf, label="z₁₁", frame=:box)
plot!(ylabel="Probability density")
plot!(title="One Latent Layer")
plot_reference_gaussian!()
p2 = histogram(z_test_data_1[2,:],normalize=:pdf, label="z₁₂", frame=:box)
plot!(ylabel="Probability density")
plot!(xlabel="Latent variable")
plot_reference_gaussian!()

p11 = histogram(z_test_data_2[1,:],normalize=:pdf, label="z₂₁", frame=:box)
plot!(ylabel="Probability density")
plot!(title="Two Latent Layers")
plot_reference_gaussian!()
p22 = histogram(z_test_data_2[2,:],normalize=:pdf, label="z₂₂", frame=:box)
plot!(ylabel="Probability density")
plot!(xlabel="Latent variable")
plot_reference_gaussian!()

p = plot(p1,p11,p2,p22, layout=(2,2), size = (900, 500), left_margin = 4mm, bottom_margin = 4mm)
savefig(p, "hvae_example_latent_comp.svg")