using Lux
using GenerativeModelProtocols
using Documenter
using DocumenterCitations

using Distances
using OptimalTransport
using Tulip

DocMeta.setdocmeta!(
    GenerativeModelProtocols, 
    :DocTestSetup, 
    :(using GenerativeModelProtocols); 
    recursive = true
)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib"); 
    style=:numeric # Options: :numeric, :authoryear, or :alphabetic
)

ext_modules = [
    GenerativeModelProtocols,
    isdefined(Base, :get_extension) ? Base.get_extension(GenerativeModelProtocols, :GenerativeModelProtocolsOptimalTransportExt) : GenerativeModelProtocols.GenerativeModelProtocolsOptimalTransportExt,
    isdefined(Base, :get_extension) ? Base.get_extension(GenerativeModelProtocols, :GenerativeModelProtocolsDistancesExt) : GenerativeModelProtocols.GenerativeModelProtocolsDistancesExt,
    isdefined(Base, :get_extension) ? Base.get_extension(GenerativeModelProtocols, :GenerativeModelProtocolsTulipExt) : GenerativeModelProtocols.GenerativeModelProtocolsTulipExt
]

makedocs(;
    plugins=[bib],
    modules = Vector{Module}(filter(!isnothing, ext_modules)),
    authors = "José Romero <jrodriguesro@umass.edu>",
    sitename = "GenerativeModelProtocols.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.io",
        edit_link = "main",
        assets=String["assets/citations.css", "assets/custom.css"], 
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Generative Models" => [
            "Variational Autoencoders" => "generative_models/variational_autoencoders.md",
            "Gaussian Mixture Models" => "generative_models/gaussian_mixture_models.md",
            "Diffusion Models" => "generative_models/diffusion_models.md",
            "Generative Adversarial Networks" => "generative_models/generative_adversarial_networks.md",
            "Normalizing Flows" => "generative_models/normalizing_flows.md"
        ],
        "Distribution Distances" => [
            "Energy Distance" => "distribution_distances/energy_distance.md",
            "Earth Mover's Distance" => "distribution_distances/earth_mover_distance.md",
            "Sinkhorn's Distance" => "distribution_distances/sinkhorn_distance.md"
        ],
        "Examples" => [
            "Getting Started" => [
                "Variational Autoencoder" => "examples/getting_started/vae/getting_started.md",
                "Gaussian Mixture Model" => "examples/getting_started/gmm/getting_started.md",
                "Diffusion Model" => "examples/getting_started/dm/getting_started.md",
                "Generative Adversarial Network" => "examples/getting_started/gan/getting_started.md",
                "Normalizing Flow" => "examples/getting_started/nf/getting_started.md"
            ],
            "β Annealing" => "examples/beta_annealing/beta_annealing.md",
            "Hierarchical VAEs" => "examples/hierarchical_vaes/hierarchical_vaes.md",
            "Comparing Models" => "examples/comparing_models/comparing_models.md",
            "Using GPUs" => "examples/using_gpus/using_gpus.md",
            "Using Reactant.jl" => "examples/using_reactant/using_reactant.md",
            "Customizing Models" => "examples/customizing_models/customizing_models.md",
            "Saving and Loading Models" => "examples/save_load/save_load.md"
        ],
        "References" => "references.md"
    ],

)

deploydocs(; 
    repo = "github.com/JoseRodriguezRomero/GenerativeModelProtocols.jl.git",
    devbranch = "main"
)
