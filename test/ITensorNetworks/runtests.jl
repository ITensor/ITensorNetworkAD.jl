using ITensorNetworkAD
using Test

@testset "ITensorNetworks.jl" begin
  for filename in ["peps.jl", "models.jl", "projectors.jl", "itensor_network.jl"]
    println("Running $filename in ITensorNetworks.jl")
    include(filename)
  end
end
