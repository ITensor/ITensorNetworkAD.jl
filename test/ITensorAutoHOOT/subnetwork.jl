using ITensors, ITensorNetworkAD
using ITensorNetworkAD.ITensorAutoHOOT: SubNetwork, get_leaves

@testset "test SubNetwork" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(4, "k")
  l = Index(5, "l")
  A = randomITensor(i)
  B = randomITensor(j)
  C = randomITensor(k)
  D = randomITensor(l)

  AB = SubNetwork([A, B])
  ABCD = SubNetwork([AB, C, D])
  @test get_leaves(ABCD) == [A, B, C, D]
  @test inds(ABCD) == [i, j, k, l]
  @test noncommoninds(ABCD, AB, C) == [l]
end
