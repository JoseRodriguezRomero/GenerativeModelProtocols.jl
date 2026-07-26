using Flux
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
    repo = "https://github.com/JoseRodriguezRomero/GenerativeModelProtocols.jl/blob/{commit}{path}#{line}",
    sitename = "GenerativeModelProtocols.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JoseRodriguezRomero.github.io/GenerativeModelProtocols.jl",
        edit_link = "main",
        assets=String["assets/citations.css"], 
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Generative Models" => [
            "Variational Autoencoders" => "generative_models/variational_autoencoders.md",
            "Gaussian Mixture Models" => "generative_models/gaussian_mixture_models.md",
            "Diffusion Models" => "generative_models/diffusion_models.md"
        ],
        "Examples" => [
            "Getting Started" => [
                "Variational Autoencoder" => "examples/getting_started/vae/getting_started.md"
                "Gaussian Mixture Model" => "examples/getting_started/gmm/getting_started.md"
            ],
            "Saving and Loading Models" => "examples/save_load/save_load.md"
        ],
    ],
)

deploydocs(; repo = "github.com/JoseRodriguezRomero/GenerativeModelProtocols.jl")

