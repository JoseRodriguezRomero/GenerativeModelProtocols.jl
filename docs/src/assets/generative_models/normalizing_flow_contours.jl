using Plots
using Plots.Measures


ρ0(x, y) = exp(-(x^2 + y^2))
ρ1(x, y) = 2 * exp(-0.3 * (2*(x - 1)^2 + (y + 0.5)^2)) + 2 * exp(-2.0 * (0.5*(x + 2)^2 + 0.8*(y - 1)^2))
ρ(x, y, t) = ρ0(x, y) + t * (ρ1(x, y) - ρ0(x, y))

x_range = range(-6, 6, length=300)
y_range = range(-6, 6, length=300)

contour_levels = collect(0.2:0.2:2.0)
contour_color = :cool

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

final_plot = plot(p1, p2, p3, 
    layout=(1, 3), 
    size=(1200, 440), 
    left_margin=4mm, right_margin=4mm, top_margin=8mm, bottom_margin=4mm
)

display(final_plot)
savefig(final_plot, "normalizing_flow_contours.svg")

