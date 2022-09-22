using ITensorNetworkAD
using Test

@testset "ITensorNetworks.jl" begin
  for filename in [
    # "lattice.jl",
    # "peps.jl",
    # "models.jl",
    # "mpstensor.jl",
    # "treetensor.jl",
    # "itensor_network.jl",
    # "projectors.jl",
    "interface.jl",
    # "tree.jl",
    # "indexgroup.jl",
  ]
    println("Running $filename in ITensorNetworks.jl")
    include(filename)
  end
end
