using ITensorNetworkAD
using Test

@testset "ITensorNetworks.jl" begin
  for filename in ["basic_ops.jl", "subnetwork.jl"]
    println("Running $filename in ITensorNetworks.jl")
    include(filename)
  end
end
