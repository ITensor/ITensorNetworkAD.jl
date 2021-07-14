using ITensorNetworkAD
using Test

@testset "ITensorNetworkAD.jl" begin
  for filename in [
    "ITensorAutoHOOT/runtests.jl",
    "ITensorChainRules/runtests.jl",
    "ITensorNetworks/runtests.jl",
  ]
    println("Running $filename")
    include(filename)
  end
end
