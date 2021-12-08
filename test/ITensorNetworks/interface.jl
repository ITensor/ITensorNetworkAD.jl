using ITensors, SweepContractor, ITensorNetworkAD
using ITensorNetworkAD.ITensorNetworks: ITensor_networks

@testset "test the interface" begin
  LTN = LabelledTensorNetwork{Char}()
  LTN['A'] = Tensor(['D', 'B'], [i^2 - 2j for i in 0:2, j in 0:2], 0, 1)
  LTN['B'] = Tensor(['A', 'D', 'C'], [-3^i * j + k for i in 0:2, j in 0:2, k in 0:2], 0, 0)
  LTN['C'] = Tensor(['B', 'D'], [j for i in 0:2, j in 0:2], 1, 0)
  LTN['D'] = Tensor(['A', 'B', 'C'], [i * j * k for i in 0:2, j in 0:2, k in 0:2], 1, 1)

  sweep = sweep_contract(LTN, 100, 100; fast=true)
  out = ldexp(sweep...)
  @test isapprox(out, contract(ITensor_networks(LTN))[])
end
