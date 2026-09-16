using Lux
using GenerativeModelProtocols
using Documenter
using DocumenterCitations

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

makedocs(;
    plugins=[bib],
    modules = [GenerativeModelProtocols],
    authors = "José Romero <jrodriguesro@umass.edu>",
    sitename = "GenerativeModelProtocols.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.io",
        edit_link = "main",
        assets=String["assets/citations.css"], 
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Generative Models" => [
            "Variational Autoencoders" => "generative_models/variational_autoencoders.md",
            "Gaussian Mixture Models" => "generative_models/gaussian_mixture_models.md",
            "Diffusion Models" => "examples/diffusion_models.md",
            "Generative Adversarial Networks" => "generative_models/generative_adversarial_networks.md"
        ],
        "Examples" => [
            "Getting Started" => [
                "Variational Autoencoder" => "examples/getting_started/vae/getting_started.md",
                "Gaussian Mixture Model" => "examples/getting_started/gmm/getting_started.md",
                "Diffusion Model" => "examples/getting_started/dm/getting_started.md",
                "Generative Adversarial Network" => "examples/getting_started/gan/getting_started.md"
            ],
            "β Annealing" => "examples/beta_annealing/beta_annealing.md",
            "Hierarchical VAEs" => "examples/hierarchical_vaes/hierarchical_vaes.md",
            "Customizing Models" => "examples/customizing_models/customizing_models.md",
            "Saving and Loading Models" => "examples/save_load/save_load.md"
        ],
        "References" => "references.md"
    ],
)

deploydocs(; 
    repo = "://github.com",
    devbranch = "main"
)
