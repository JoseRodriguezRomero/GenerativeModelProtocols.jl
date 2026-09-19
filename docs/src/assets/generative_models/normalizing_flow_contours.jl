using Plots
using Plots.Measures

using Optimisers, Lux
using DifferentialEquations
using GenerativeModelProtocols

using FileIO, HDF5

normal(σ, μ, x) = exp(-((x-μ)^2)/(2*σ^2)) / sqrt(2*π*σ^2)

protocol = GenerativeModelProtocol("sample_nf_model.h5")

ρ0(x, y) = normal(1.0, 0.0, x) * normal(1.0, 0.0, y)
ρ1(x, y) = 
    (1.25 / 3.0) * (normal(0.5,  1.0, x) * normal(1.0, -0.5, y)) +
    (1.75 / 3.0) * (normal(1.0, -3.0, x) * normal(0.6,  1.0, y))

ρ(x, y, t) = ρ0(x, y) + t * (ρ1(x, y) - ρ0(x, y))

x_range = range(-6, 6, length=300)
y_range = range(-6, 6, length=300)

c1 = 0.02
c2 = 0.2

contour_levels = collect(c1:((c2-c1)/6.0):c2)
contour_color = :cool

## Make Plots
t1_samples = protocol(100)
t0_samples = encode(protocol, t1_samples)
t_half_samples = decode(protocol, t0_samples; t_final = 0.5)

plot_lims = (-5, 5)

z0 = [ρ0(x, y) for y in y_range, x in x_range]
p1 = contour(x_range, y_range, z0,
    levels=contour_levels,
    color=contour_color,
    linewidth=1.2,
    colorbar=false,
    title="Latent Space\n(t = 0)\n",
    xlims=plot_lims, ylims=plot_lims,
    grid=true, gridalpha=0.15, gridstyle=:solid,
    xaxis=true, yaxis=true,
    xformatter = _ -> "",
    yformatter = _ -> "",
    aspect_ratio=:equal,
    frame=:box
)
scatter!(t0_samples[1,:],t0_samples[2,:], legend = false)

z_mid = [ρ(x, y, 0.5) for y in y_range, x in x_range]
p2 = contour(x_range, y_range, z_mid,
    levels=contour_levels,
    color=contour_color,
    linewidth=1.2,
    colorbar=false,
    title="Intermediate Latent Space\n(t = 0.5)\n",
    xlims=plot_lims, ylims=plot_lims,
    grid=true, gridalpha=0.15, gridstyle=:solid,
    xaxis=true, yaxis=true,
    xformatter = _ -> "",
    yformatter = _ -> "",
    aspect_ratio=:equal,
    frame=:box
)
scatter!(t_half_samples[1,:],t_half_samples[2,:], legend = false)

z1 = [ρ1(x, y) for y in y_range, x in x_range]
p3 = contour(x_range, y_range, z1,
    levels=contour_levels,
    color=contour_color,
    linewidth=1.2,
    colorbar=false,
    title="Target Space\n(t = 1)\n",
    xlims=plot_lims, ylims=plot_lims,
    grid=true, gridalpha=0.15, gridstyle=:solid,
    xaxis=true, yaxis=true,
    xformatter = _ -> "",
    yformatter = _ -> "",
    aspect_ratio=:equal,
    frame=:box
)
scatter!(t1_samples[1,:],t1_samples[2,:], legend = false)

final_plot = plot(p1, p2, p3, 
    layout=(1, 3), 
    size=(1200, 440), 
    left_margin=4mm, right_margin=4mm, top_margin=8mm, bottom_margin=4mm
)

display(final_plot)
savefig(final_plot, "normalizing_flow_contours.svg")
