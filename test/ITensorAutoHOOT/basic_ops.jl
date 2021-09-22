using ITensorNetworkAD
using AutoHOOT
using ITensors
using Zygote
using ITensorNetworkAD.ITensorAutoHOOT: SubNetwork

const go = AutoHOOT.graphops
const itensorah = ITensorNetworkAD.ITensorAutoHOOT

@testset "test interface" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  l = Index(4, "l")
  m = Index(5, "m")

  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, l)
  D = randomITensor(l, m)

  out = A * B * C * D
  network = [A, C, B, D]
  nodes, dict = itensorah.generate_einsum_expr([network])
  network = itensorah.extract_network(nodes[1], dict)
  out2 = contract(network)

  AB = SubNetwork([A, B])
  ABC = SubNetwork([AB, C])
  ABCD = SubNetwork([D, ABC])
  nodes, dict = itensorah.generate_einsum_expr([ABCD])
  network = itensorah.extract_network(nodes[1], dict)
  out3 = contract(network)

  @test isapprox(out, out2)
  @test isapprox(out, out3)
end

@testset "test compute" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(4, "k")
  l = Index(5, "l")
  m = Index(6, "m")

  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, l)
  D = randomITensor(l, m)
  E = randomITensor(m, i)

  out = A * B * C * D * E
  network = [A, B, C, D, E]

  nodes, dict = itensorah.generate_einsum_expr([network])
  node = go.generate_optimal_tree(nodes[1])
  out_list = itensorah.compute_graph([node], dict)
  out2 = out_list[1]

  @test isapprox(out, out2)
end

@testset "test optimal contraction path" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(4, "k")
  l = Index(5, "l")
  m = Index(6, "m")

  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, l)
  D = randomITensor(l, m)
  E = randomITensor(m, i)

  out = A * B * C * D * E

  network = itensorah.generate_optimal_tree([A, B, C, D, E])
  out2 = contract(network)

  @test isapprox(out, out2)
end

@testset "test gradient" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(4, "k")
  l = Index(5, "l")
  m = Index(6, "m")

  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, l)
  D = randomITensor(l, m)
  E = randomITensor(m, i)

  gradA_direct = B * C * D * E
  gradB_direct = A * C * D * E

  networks = itensorah.gradients([A, B, C, D, E], [A, B])
  gradA = contract(networks[1])
  gradB = contract(networks[2])

  @test isapprox(gradA_direct, gradA)
  @test isapprox(gradB_direct, gradB)
end

@testset "test batch_tensor_contraction" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, i)

  function network(A)
    tensor_network = [A, B, C]
    out = itensorah.batch_tensor_contraction([tensor_network], A)
    return sum(out)[]
  end
  function tree_network(A)
    tensor_network = SubNetwork(SubNetwork(A, B), C)
    out = itensorah.batch_tensor_contraction([tensor_network], A)
    return sum(out)[]
  end
  grad_A = gradient(network, A)
  tree_grad_A = gradient(tree_network, A)
  @test isapprox(grad_A[1], B * C)
  @test isapprox(tree_grad_A[1], B * C)
end

@testset "test zygote interface for inner product" begin
  i = Index(2, "i")
  a = randomITensor(i)
  # build a symmetric H
  H = ITensor(i, i')
  H[i => 1, i' => 1] = 1.0
  H[i => 2, i' => 1] = 2.0
  H[i => 1, i' => 2] = 2.0
  H[i => 2, i' => 2] = 3.0

  function inner(a)
    b = prime(a)
    network = [a, H, b]
    inner = itensorah.batch_tensor_contraction([network], network...)
    return sum(inner)[]
  end
  grad = gradient(inner, a)
  @test isapprox((grad[1])', 2 * H * a)
end

@testset "test zygote interface with sum" begin
  A = ITensor(3.0)
  B = ITensor(2.0)
  function add(A, B)
    return sum([A, B])[]
  end
  grad = gradient(add, A, B)
  @test isapprox(grad[1], ITensor(1.0))
end

@testset "test zygote interface with multiple networks" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  A = randomITensor(i, j)
  B = randomITensor(j, i)
  C = randomITensor(j, k)
  D = randomITensor(k, l)
  E = randomITensor(l, i)

  function inner(A)
    networks = [[A, B], [A, C, D, E]]
    contract = itensorah.batch_tensor_contraction(networks, A, B, C, D, E)
    return sum(contract)[]
  end
  grad = gradient(inner, A)
  @test isapprox(grad[1], B + C * D * E)
end

@testset "test simple hvp" begin
  i = Index(2, "i")
  A = randomITensor(i)
  B = randomITensor(i, i')
  B[i => 1, i' => 1] = 1.0
  B[i => 2, i' => 1] = 2.0
  B[i => 1, i' => 2] = 2.0
  B[i => 2, i' => 2] = 3.0
  v = randomITensor(i)

  network(x) = ((x * B) * prime(x))[]
  grad(x) = gradient(network, x)[1]
  inner(x) = (grad(x) * v)[]
  hvp(x) = gradient(inner, x)[1]

  hvp_out = hvp(A)
  hvp_true = noprime(2 * B * v)
  @test isapprox(hvp_out, hvp_true)
end

@testset "test hvp with batch_tensor_contraction" begin
  i = Index(2, "i")
  A = randomITensor(i)
  B = randomITensor(i, i')
  B[i => 1, i' => 1] = 1.0
  B[i => 2, i' => 1] = 2.0
  B[i => 1, i' => 2] = 2.0
  B[i => 2, i' => 2] = 3.0
  v = randomITensor(i)

  function network(x)
    xprime = prime(x)
    inner = sum(itensorah.batch_tensor_contraction([[x, xprime, B]], x, xprime))
    return inner[]
  end
  grad(x) = gradient(network, x)[1]
  inner(x) = (grad(x) * v)[]
  hvp(x) = gradient(inner, x)[1]
  hvp_out = hvp(A)
  hvp_true = noprime(2 * B * v)
  @test isapprox(hvp_out, hvp_true)
end
