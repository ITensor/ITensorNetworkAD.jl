using ITensorNetworkAD
using Test

@testset "ITensorNetworks.jl" begin
  for filename in [
    "lattice.jl",
    "subnetwork.jl",
    "peps.jl",
    "models.jl",
    "itensor_network.jl",
    "projectors.jl",
  ]
    println("Running $filename in ITensorNetworks.jl")
    include(filename)
  end
end
