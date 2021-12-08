using ITensors, SweepContractor, ITensorNetworkAD
using ITensorNetworkAD.ITensorNetworks: ITensor_networks, line_network, TreeTensor
using ITensorNetworkAD.ITensorAutoHOOT: SubNetwork, batch_tensor_contraction

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

@testset "test on 2D grid" begin
  ITensors.set_warn_order(100)
  L, d = 8, 2
  LTN = LabelledTensorNetwork{Tuple{Int,Int}}()
  for i in 1:L, j in 1:L
    adj = Tuple{Int,Int}[]
    i > 1 && push!(adj, (i - 1, j))
    j > 1 && push!(adj, (i, j - 1))
    i < L && push!(adj, (i + 1, j))
    j < L && push!(adj, (i, j + 1))
    LTN[i, j] = Tensor(adj, randn(d * ones(Int, length(adj))...), i, j)
  end
  inetwork = line_network(ITensor_networks(LTN))
  for rank in d .^ (1:6)
    sweep = sweep_contract(LTN, rank, rank; fast=true)
    println("rank=$rank:\t", ldexp(sweep...))
    out2 = batch_tensor_contraction(TreeTensor, [inetwork]; cutoff=1e-15, maxdim=rank)
    print("tree network", ITensor(out2[1])[], "\n")
  end
end
