using ITensorNetworkAD
using Documenter

DocMeta.setdocmeta!(
  ITensorNetworkAD, :DocTestSetup, :(using ITensorNetworkAD); recursive=true
)

makedocs(;
  modules=[ITensorNetworkAD],
  authors=["Matthew Fishman <mfishman@flatironinstitute.org>", "Linjian Ma <lma16@illinois.edu>"],
  repo="https://github.com/itensor/ITensorNetworkAD.jl/blob/{commit}{path}#{line}",
  sitename="ITensorNetworkAD.jl",
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://itensor.github.io/ITensorNetworkAD.jl",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/itensor/ITensorNetworkAD.jl", devbranch="main")
