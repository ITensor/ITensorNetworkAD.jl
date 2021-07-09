using ITensorNetworkAD
using Documenter

DocMeta.setdocmeta!(ITensorNetworkAD, :DocTestSetup, :(using ITensorNetworkAD); recursive=true)

makedocs(;
    modules=[ITensorNetworkAD],
    authors="Matthew Fishman <mfishman@flatironinstitute.org> and contributors",
    repo="https://github.com/mtfishman/ITensorNetworkAD.jl/blob/{commit}{path}#{line}",
    sitename="ITensorNetworkAD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mtfishman.github.io/ITensorNetworkAD.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mtfishman/ITensorNetworkAD.jl",
    devbranch="main",
)
