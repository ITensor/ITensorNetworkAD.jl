using Test

@testset "ITensorChainRules" begin
  for filename in ["chainrules.jl", "optimization.jl"]
    include(filename)
  end
end
