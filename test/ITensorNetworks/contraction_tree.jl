using ITensors, ITensorNetworkAD
using ITensorNetworkAD.ITensorNetworks: ContractNode, get_leaves

@testset "test ContractNode" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(4, "k")
  l = Index(5, "l")
  A = randomITensor(i)
  B = randomITensor(j)
  C = randomITensor(k)
  D = randomITensor(l)

  AB = ContractNode([A, B])
  ABCD = ContractNode([AB, C, D])
  @test get_leaves(ABCD) == [A, B, C, D]
  @test inds(ABCD) == [i, j, k, l]
  @test noncommoninds(ABCD, AB, C) == [l]
end
