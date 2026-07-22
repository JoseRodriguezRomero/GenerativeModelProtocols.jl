using GenerativeModelProtocols
using Documenter

DocMeta.setdocmeta!(
    GenerativeModelProtocols, 
    :DocTestSetup, 
    :(using GenerativeModelProtocols); 
    recursive = true
)

makedocs(;
    modules = [GenerativeModelProtocols],
    authors = "José Romero <jrodriguesro@umass.edu>",
    repo = "https://github.com/JoseRodriguezRomero/GenerativeModelProtocols.jl/blob/{commit}{path}#{line}",
    sitename = "GenerativeModelProtocols.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JoseRodriguezRomero.github.io/GenerativeModelProtocols.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Generative Models" => [
            "Variational Autoencoders" => "generative_models/variational_autoencoders.md",
            "Diffusion Models" => "generative_models/diffusion_models.md"
        ],
        "Examples" => [
            "Variational Autoencoders" => "examples/variational_autoencoders.md",
            "Diffusion Models" => "examples/diffusion_models.md"
        ]
    ],
)

deploydocs(; repo = "github.com/JoseRodriguezRomero/GenerativeModelProtocols.jl")

