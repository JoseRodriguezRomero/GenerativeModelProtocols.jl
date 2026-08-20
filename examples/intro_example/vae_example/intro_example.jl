using GenerativeModelProtocols
using Flux, Plots
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

model = GenerativeModelProtocols.VariationalAutoencoder(2, 2)
protocol = GenerativeModelProtocol(model, train_data;
    batchsize = 256,
    epochs = 1500,
    optimiser = Adam(; eta = 1.0E-4, beta = (0.95,0.999)),
    device = cpu_device()
)
train!(protocol; β = 0.2)

synthetic_data = protocol(num_samples)

p1 = scatter(x_train, y_train, title="Training data", label=false,frame=:box)
p2 = scatter(synthetic_data[1,:], synthetic_data[2,:], title="Synthetic data", label=false,frame=:box)

function plot_model_density(protocol)
    data = protocol(50000)
    
    fig = histogram2d(data[1,:], data[2,:],
        bins=(200,200),
        show_empty_bins=true,
        title="VAE PDF\n2D Histogram",
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
p22 = plot_model_density(protocol)

p = plot(p1,p11,p2,p22; layout=(2,2), size = (900, 800), link = :x)
savefig(p, "intro_example.svg")

## Test reconstruction
x_test, y_test = make_data(800)
test_data = collect(transpose(hcat(x_test,y_test)))
z_test_data = GenerativeModelProtocols.encode(protocol,test_data)
recon_test_data = GenerativeModelProtocols.decode(protocol,z_test_data)

p = scatter(x_test, y_test, label="input",frame=:box)
scatter!(recon_test_data[1,:], recon_test_data[2,:], label="reconstruction", frame=:box, legend = :top)
savefig(p, "intro_example_recon.svg")

## Test latent space
x_test, y_test = make_data(10000)
test_data = collect(transpose(hcat(x_test,y_test)))
z_test_data = GenerativeModelProtocols.encode(protocol,test_data)

function reference_gaussian(x, μ, σ²)
    return exp(-((x - μ)^2)/(2.0 * σ²))/sqrt(2.0*π*σ²)
end

function plot_reference_gaussian!()
    x = collect(-10.0:0.01:10)
    return plot!(x, reference_gaussian.(x,0,1), style=:dash, label=L"\mathcal{N} (0,1)", linewidth=5)
end

p1 = histogram(z_test_data[1,:],normalize=:pdf, label="z₁", frame=:box)
plot!(ylabel="Probability density")
plot_reference_gaussian!()
p2 = histogram(z_test_data[2,:],normalize=:pdf, label="z₂", frame=:box)
plot!(ylabel="Probability density")
plot!(xlabel="Latent variable")
plot_reference_gaussian!()

p = plot(p1,p2, layout=(2,1))
savefig(p, "intro_example_latents.svg")

## Test latent space independence
using StatsBase, Plots.PlotMeasures

pearson_corr = cor(z_test_data[1,:],z_test_data[2,:])
spearman_corr = corspearman(z_test_data[1,:],z_test_data[2,:])

p = scatter(z_test_data[1,:],z_test_data[2,:], label = false, frame=:box)
plot!(xlabel="z₁", ylabel="z₂")
plot!(title="pearson_corr = $(round(pearson_corr,digits=4))\nspearman_corr = $(round(spearman_corr,digits=4))")
plot!(top_margin = 5mm)
savefig(p, "intro_example_latents_corr.svg")

