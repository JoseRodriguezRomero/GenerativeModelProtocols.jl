using Plots
using StatsBase
using Plots.Measures

## Distribution Distance Comparisons

function read_file(file_name::String)
    lines = readlines(file_name)[2:end]

    samples = zeros(Float64, length(lines))
    μ = zeros(Float64, length(lines))
    σ = zeros(Float64, length(lines))

    for i in eachindex(lines)
        line = split(lines[i])
        samples[i] = parse(Float64, line[1])
        μ[i] = parse(Float64, line[2])
        σ[i] = parse(Float64, line[3])
    end

    return samples, μ, σ
end

function read_data(base_dir::String, xlabel, ylabel)

    model_files = [
        ("ref_model.txt", "Reference"),
        ("dm_model.txt", "Diffusion Model"),
        ("nf_model.txt", "Normalizing Flow"),
        ("gmm_model.txt", "Gaussian Mixture Model"),
        ("vae_model.txt", "Variational Autoencoder"),
        ("gan_model.txt", "Generative Adversarial Network")
    ]

    data = []

    for (file, label) in model_files
        samples, μ, σ = read_file(joinpath(base_dir, file))
        push!(data, (samples, μ, σ, label))
    end

    sample_counts = data[1][1]

    for (samples, _, _, label) in data
        if samples != sample_counts
            error("Sample counts for $label do not match the Reference model.")
        end
    end

    n_groups = length(sample_counts)
    n_models = length(data)

    group_width = 0.8
    bar_width = group_width / n_models
    group_centers = collect(1:n_groups)

    p = plot(
        xlabel = xlabel,
        ylabel = ylabel,
        xticks = (
            group_centers,
            string.(Int.(sample_counts))
        ),
        legend = false,
        size = (500, 500),
        bottom_margin = 5mm,
        frame = :box
    )

    for model_idx in 1:n_models
        _, means, _, _ = data[model_idx]

        x = group_centers .-
            group_width / 2 .+
            (model_idx - 0.5) * bar_width

        bar!(
            p,
            x,
            means,
            bar_width = bar_width,
            label = nothing,
            linewidth = 0,
        )
    end

    for model_idx in 1:n_models
        _, means, errors, _ = data[model_idx]

        x = group_centers .-
            group_width / 2 .+
            (model_idx - 0.5) * bar_width

        scatter!(
            p,
            x,
            means,
            yerror = errors,
            marker = nothing,
            label = false,
            color = :black,
            linecolor = :black,
            linewidth = 1.5,
        )
    end

    return p
end

p1 = read_data("emd_dist_comp/", "", "𝔼[Earth Mover's Distance]")
plot!(p1, ylims = [0, 0.4])

p2 = read_data("sinkhorn_dist_comp/", "", "𝔼[Sinkhorn Distance]")
plot!(p2, ylims = [0, 0.25], yticks = 0:0.05:0.25)

p3 = read_data("energy_dist_comp/", "Number of Samples", "𝔼[Energy Distance]")
plot!(p3, ylims = [-0.005, 0.01])

labels = [
    "Reference",
    "Diffusion Model",
    "Normalizing Flow",
    "Gaussian Mixture Model",
    "Variational Autoencoder",
    "Generative Adversarial Network"
]

colors = palette(:auto)

legend_plot = plot(
    legend = :top,
    legend_columns = 3,
    framestyle = :none,
    axis = nothing,
    grid = false,
    size = (200, 1000),
)

for (i, label) in enumerate(labels)
    plot!(
        legend_plot,
        [NaN],
        [NaN],
        label = label,
        color = colors[i],
        linewidth = 8,
    )
end

p = plot(
    legend_plot,
    p1,
    p2,
    p3,
    layout = @layout([
        a{0.08h}
        b
        c
        d
    ]),
    size = (900, 800),
    bottom_margin = 1mm,
    left_margin = 2mm,
)

savefig(p, "model_distance_comparisons.svg")

## Training Time Comparisons

model_keys, train_times = open("training_timings.txt", "r") do file
    lines = readlines(file)
    [split(l)[1] for l in lines], [parse(Float64, split(l)[2]) for l in lines]
end

label_map = Dict(
    "dm_model" => "Diffusion Model",
    "nf_model" => "Normalizing Flow",
    "gmm_model" => "Gaussian\nMixture Model",
    "vae_model" => "Variational\nAutoencoder",
    "gan_model" => "Generative\nAdversarial Network"
)

labels = [label_map[model_key] for model_key in model_keys]
y = collect(1:length(labels))

p = bar(
    [label_map[model_key] for model_key in model_keys],
    train_times,
    permute = (:x, :y),
    xlabel = "Training Time (s)",
    ylabel = "Model",
    legend = false,
    size = (700, 450),
    left_margin = 3mm,
    bottom_margin = 3mm,
    frame = :box
)

p = bar(
    [label_map[model_key] for model_key in model_keys],
    train_times ./ 60.0,
    permute = (:x, :y),
    ylabel = "Training Time (minutes)",
    xlabel = "Model",
    legend = false,
    size = (700, 450),
    left_margin = 2mm,
    bottom_margin = 2mm,
    frame = :box,
    xlims = (-0.15, length(model_keys) + 0.15),
)

savefig(p, "model_training_time_comparisons.svg")

## Inference Time Comparisons

using GenerativeModelProtocols
using FileIO, HDF5, StatsBase
using DifferentialEquations

dm_protocol = GenerativeModelProtocol("trained_models/dm_model.h5")
nf_protocol = GenerativeModelProtocol("trained_models/nf_model.h5")
gmm_protocol = GenerativeModelProtocol("trained_models/gmm_model.h5")
vae_protocol = GenerativeModelProtocol("trained_models/vae_model.h5")
gan_protocol = GenerativeModelProtocol("trained_models/gan_model.h5")

function measure_inference_time(protocol::GenerativeModelProtocol, num_samples::Int)
    start_time = time()
    protocol(num_samples)
    end_time = time()
    return end_time - start_time
end

function measure_inference_times(protocol::GenerativeModelProtocol, num_samples::Int, num_reps::Int)
    protocol(num_samples) # ensure inference is precompiled

    return [measure_inference_time(protocol, num_samples) for _ in 1:num_reps]
end

function measure_inference_times(protocol::GenerativeModelProtocol, num_samples::Vector{Int}, num_reps::Int)
    inference_mean_time = zeros(Float64, length(num_samples))
    inference_std_time = zeros(Float64, length(num_samples))

    for (i, samples) in enumerate(num_samples)
        times = measure_inference_times(protocol, samples, num_reps)
        inference_mean_time[i] = mean(times)
        inference_std_time[i] = std(times)
    end

    return inference_mean_time, inference_std_time
end

num_samples = collect(100:100:200)

dm_inference_mean, dm_inference_std = measure_inference_times(dm_protocol, num_samples, 200)
nf_inference_mean, nf_inference_std = measure_inference_times(nf_protocol, num_samples, 200)
gmm_inference_mean, gmm_inference_std = measure_inference_times(gmm_protocol, num_samples, 200)
vae_inference_mean, vae_inference_std = measure_inference_times(vae_protocol, num_samples, 200)
gan_inference_mean, gan_inference_std = measure_inference_times(gan_protocol, num_samples, 200)

inference_means = [
    dm_inference_mean,
    nf_inference_mean,
    gmm_inference_mean,
    vae_inference_mean,
    gan_inference_mean
]

inference_stds = [
    dm_inference_std,
    nf_inference_std,
    gmm_inference_std,
    vae_inference_std,
    gan_inference_std
]

times_keys = [
    "Diffusion Model",
    "Normalizing Flow",
    "Gaussian Mixture Model",
    "Variational Autoencoder",
    "Generative Adversarial Network"
]

using Printf

function print_inference_times(num_samples, times_keys, inference_means, inference_stds)
    println("| Model | " * join(["$n samples" for n in num_samples], " | ") * " |")
    println("|:---|" * join(fill("---:", length(num_samples)), "|") * "|")

    for (i, model) in enumerate(times_keys)
        values = [
            @sprintf("%.2e ± %.2e", inference_means[i][j] * 1000,
                     inference_stds[i][j] * 1000)
            for j in eachindex(num_samples)
        ]

        println("| $model | " * join(values, " | ") * " |")
    end

    println("\n*Inference time in milliseconds; values are mean ± standard deviation.*")
end

inference_means = [
    dm_inference_mean,
    nf_inference_mean,
    gmm_inference_mean,
    vae_inference_mean,
    gan_inference_mean,
]

inference_stds = [
    dm_inference_std,
    nf_inference_std,
    gmm_inference_std,
    vae_inference_std,
    gan_inference_std,
]

print_inference_times(num_samples, times_keys, inference_means, inference_stds)

