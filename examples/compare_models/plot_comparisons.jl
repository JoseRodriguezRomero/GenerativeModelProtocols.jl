using Plots
using StatsBase
using Plots.Measures

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
plot!(p2, ylims = [0, 0.25])

p3 = read_data("energy_dist_comp/", "Number of Samples", "𝔼[Energy Distance]")
plot!(p3, ylims = [-0.01, 0.02])

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

plot(
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

