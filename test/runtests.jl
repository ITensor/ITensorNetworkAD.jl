using ITensorNetworkAD
using Test

@testset "ITensorNetworkAD.jl" begin
  include("ITensorNetworks/runtests.jl")
  include("ITensorChainRules/runtests.jl")
  include("ITensorAutoHOOT/runtests.jl")
end
