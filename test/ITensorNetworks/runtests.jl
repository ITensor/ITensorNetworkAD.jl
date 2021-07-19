using ITensorNetworkAD
using Test

@testset "ITensorNetworks.jl" begin
  for filename in ["models.jl", "projectors.jl", "itensor_network.jl"]
    println("Running $filename in ITensorNetworks.jl")
    include(filename)
  end
end
