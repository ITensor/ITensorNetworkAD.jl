using ITensors, Random, SweepContractor, ITensorNetworkAD
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
  Random.seed!(1234)
  ITensors.set_warn_order(100)
  row, column, d = 8, 8, 2
  LTN = LabelledTensorNetwork{Tuple{Int,Int}}()
  for i in 1:row, j in 1:column
    adj = Tuple{Int,Int}[]
    i > 1 && push!(adj, (i - 1, j))
    j > 1 && push!(adj, (i, j - 1))
    i < row && push!(adj, (i + 1, j))
    j < column && push!(adj, (i, j + 1))
    LTN[i, j] = Tensor(adj, randn(d * ones(Int, length(adj))...), i, j)
  end
  tnet = ITensor_networks(LTN)
  element_grouping = line_network(tnet)
  tnet_mat = reshape(tnet, row, column)
  line_grouping = SubNetwork(tnet_mat[:, 1])
  for i in 2:column
    line_grouping = SubNetwork(line_grouping, tnet_mat[:, i]...)
  end

  function get_contracted_peps(rank)
    x = tnet_mat[:, 1]
    for i in 2:(column - 1)
      A = tnet_mat[:, i]
      x = contract(MPO(A), MPS(x); cutoff=1e-15, maxdim=rank)[:]
    end
    out_mps = contract(x..., tnet_mat[:, column]...)
    sweep = sweep_contract(LTN, rank, rank; fast=true)
    out = ldexp(sweep...)
    out2 = batch_tensor_contraction(
      TreeTensor, [element_grouping]; cutoff=1e-15, maxdim=rank
    )
    out3 = batch_tensor_contraction(
      TreeTensor, [line_grouping]; cutoff=1e-15, maxdim=rank, optimize=false
    )
    return out, ITensor(out2[1])[], ITensor(out3[1])[], out_mps[]
  end

  out_true, out_element, out_line, out_mps = get_contracted_peps(d^(Int(row / 2)))
  @test abs((out_true - out_element) / out_true) < 1e-3
  @test abs((out_true - out_line) / out_true) < 1e-3
  @test abs((out_true - out_mps) / out_true) < 1e-3
  for rank in [1, 2, 3, 4, 6, 8, 10, 12, 14, 15, 16]
    out, out_element, out_line, out_mps = get_contracted_peps(rank)
    error_sweepcontractor = abs((out - out_true) / out_true)
    error_element = abs((out_element - out_true) / out_true)
    error_line = abs((out_line - out_true) / out_true)
    error_mps = abs((out_mps - out_true) / out_true)
    print(
      "maxdim, ",
      rank,
      ", error_sweepcontractor, ",
      error_sweepcontractor,
      ", error_element, ",
      error_element,
      ", error_line, ",
      error_line,
      ", error_mps, ",
      error_mps,
      "\n",
    )
  end
end
