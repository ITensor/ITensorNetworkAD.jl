using ITensors, Random, SweepContractor, ITensorNetworkAD
using ITensorNetworkAD.Profiler
using ITensorNetworkAD.ITensorNetworks: ITensor_networks, line_network, TreeTensor
using ITensorNetworkAD.ITensorAutoHOOT: SubNetwork, batch_tensor_contraction

include("utils.jl")

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

function lattice(row, column, d)
  LTN = LabelledTensorNetwork{Tuple{Int,Int}}()
  for i in 1:row, j in 1:column
    adj = Tuple{Int,Int}[]
    i > 1 && push!(adj, (i - 1, j))
    j > 1 && push!(adj, (i, j - 1))
    i < row && push!(adj, (i + 1, j))
    j < column && push!(adj, (i, j + 1))
    LTN[i, j] = Tensor(adj, randn(d * ones(Int, length(adj))...), i, j)
  end
  return LTN
end

function get_contracted_peps(LTN, rank, N)
  tnet = ITensor_networks(LTN)
  tnet_mat = reshape(tnet, N...)
  out_mps = peps_contraction_mpomps(tnet_mat; cutoff=1e-15, maxdim=rank)
  out = contract_w_sweep(LTN, rank)
  out2 = contract_element_group(tnet, rank)
  out3 = contract_line_group(tnet_mat, rank, N)
  return out, ITensor(out2[1])[], ITensor(out3[1])[], out_mps[]
end

@testset "test on 2D grid" begin
  Random.seed!(1234)
  ITensors.set_warn_order(100)
  row, column, d = 8, 8, 2
  LTN = lattice(row, column, d)

  out_true, out_element, out_line, out_mps = get_contracted_peps(
    LTN, d^(Int(row / 2)), [row, column]
  )
  @test abs((out_true - out_element) / out_true) < 1e-3
  @test abs((out_true - out_line) / out_true) < 1e-3
  @test abs((out_true - out_mps) / out_true) < 1e-3
  for rank in [1, 2, 3, 4, 6, 8, 10, 12, 14, 15, 16]
    out, out_element, out_line, out_mps = get_contracted_peps(LTN, rank, [row, column])
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

@testset "benchmark on 2D grid" begin
  Random.seed!(1234)
  ITensors.set_warn_order(100)
  row, column, d, rank = 8, 8, 10, 60
  LTN = lattice(row, column, d)
  # warm-up
  get_contracted_peps(LTN, rank, [row, column])
  @info "start benchmark"
  do_profile(true)
  for _ in 1:3
    get_contracted_peps(LTN, rank, [row, column])
  end
  profile_exit()
end
