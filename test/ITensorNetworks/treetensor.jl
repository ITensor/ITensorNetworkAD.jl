using ITensorNetworkAD
using AutoHOOT, ITensors, Zygote
using ITensorNetworkAD.ITensorNetworks:
  TreeTensor, uncontract_inds_binary_tree, tree_approximation, mincut_inds_binary_tree
using ITensorNetworkAD.ITensorNetworks: inds_network, project_boundary, Models
using ITensorNetworkAD.ITensorAutoHOOT: SubNetwork, batch_tensor_contraction

const itensorah = ITensorNetworkAD.ITensorAutoHOOT

@testset "test TreeTensor" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  l = Index(4, "l")
  m = Index(5, "m")

  A = randomITensor(i, j, k)
  B = randomITensor(k, l, m)
  C = randomITensor(i, j, l, m)
  tree_A = TreeTensor(A)
  tree_B = TreeTensor(B)
  tree_C = TreeTensor(C)

  out = A * B
  network = [tree_A, tree_B]
  nodes, dict = itensorah.generate_einsum_expr([network])
  out_list = itensorah.compute_graph(nodes, dict; cutoff=1e-15, maxdim=1000)
  @test isapprox(out, ITensor(out_list[1]))

  out = A * B * C
  out2 = contract(tree_A, tree_B, tree_C; cutoff=1e-15, maxdim=1000)
  @test isapprox(out, ITensor(out2))
end

@testset "test batch_tensor_contraction" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, l)
  D = randomITensor(l, m)
  E = randomITensor(m, i)

  function network(A)
    tensor_network = SubNetwork(SubNetwork(A, B, C), D, E)
    out = itensorah.batch_tensor_contraction(
      TreeTensor, [tensor_network], A; cutoff=1e-15, maxdim=1000, optimize=false
    )
    return sum(out)[]
  end
  grad_A = gradient(network, A)
  @test isapprox(grad_A[1], B * C * D * E)
end

@testset "test uncontract_inds_binary_tree" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  A = randomITensor(i)
  B = randomITensor(j)
  C = randomITensor(k)
  D = randomITensor(l)
  E = randomITensor(m)

  path = [[[A, B], [C, D]], E]
  uncontract_inds = [i, j, k, l, m]
  btree = uncontract_inds_binary_tree(path, uncontract_inds)
  @test btree == [[[[i], [j]], [[k], [l]]], [m]]
  out = tree_approximation([A, B, C, D, E], btree)
  @test isapprox(contract(out), A * B * C * D * E)
end

@testset "test MPS times MPO" begin
  N = (10, 3)
  linkdim = 3
  cutoff = 1e-15
  tn_inds = inds_network(N...; linkdims=linkdim)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  state = 1
  tn = project_boundary(tn, state)
  x, A = tn[:, 1], tn[:, 2]
  out_true = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=linkdim * linkdim)
  out2 = batch_tensor_contraction(
    TreeTensor,
    [SubNetwork(SubNetwork(x), SubNetwork(A))];
    cutoff=cutoff,
    maxdim=linkdim * linkdim,
  )
  tsr_true = contract(out_true...)
  tsr_nrmsquare = (tsr_true * tsr_true)[1]
  @test isapprox(tsr_true, ITensor(out2[1]))

  maxdims = [2, 4, 6, 8]
  for dim in maxdims
    out = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=dim)
    out2 = batch_tensor_contraction(
      TreeTensor, [SubNetwork(SubNetwork(x), SubNetwork(A))]; cutoff=cutoff, maxdim=dim
    )
    residual1 = tsr_true - contract(out...)
    residual2 = tsr_true - ITensor(out2[1])
    error1 = sqrt((residual1 * residual1)[1] / tsr_nrmsquare)
    error2 = sqrt((residual2 * residual2)[1] / tsr_nrmsquare)
    print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
  end
end

@testset "test mincut_inds_binary_tree" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  l = Index(4, "l")
  m = Index(5, "m")

  T = randomITensor(i, j, k, l, m)
  M = MPS(T, (i, j, k, l, m); cutoff=1e-5, maxdim=5)
  network = M[:]

  out = mincut_inds_binary_tree(network, [i, j, k, l, m])
  @test length(out) == 2
end

@testset "test PEPS" begin
  N = (8, 8) #(12, 12)
  linkdim = 2
  cutoff = 1e-15
  tn_inds = inds_network(N...; linkdims=linkdim)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  state = 1
  tn = project_boundary(tn, state)

  function get_contracted_peps(maxdim)
    x = tn[:, 1]
    for i in 2:(N[2] - 1)
      A = tn[:, i]
      x = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=maxdim)[:]
    end
    out = contract(x..., tn[:, N[2]]...)
    network = SubNetwork(tn[:, 1])
    for i in 2:(N[2])
      network = SubNetwork(network, SubNetwork(tn[:, i]))
    end
    out2 = batch_tensor_contraction(TreeTensor, [network]; cutoff=cutoff, maxdim=maxdim)
    return out[], ITensor(out2[1])[]
  end

  ITensors.set_warn_order(100)
  out_true, out2 = get_contracted_peps(linkdim^N[2])
  print(out_true, out2)
  @test abs((out_true - out2) / out_true) < 1e-3

  maxdims = [10, 11, 12, 13, 14, 15, 16] #[2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
  for dim in maxdims
    out, out2 = get_contracted_peps(dim)
    error1 = abs((out - out_true) / out_true)
    error2 = abs((out2 - out_true) / out_true)
    print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
  end
end
