using ITensors, ITensorNetworkAD, AutoHOOT, Zygote, OptimKit
using ITensorNetworkAD.ITensorNetworks: PEPS, inner_network, Models, extract_data
using ITensorNetworkAD.Optimizations: gradient_descent, generate_inner_network
using ITensorNetworkAD.ITensorAutoHOOT: batch_tensor_contraction

@testset "test monotonic loss decrease of optimization" begin
  Nx, Ny = 2, 3
  num_sweeps = 20
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=10)
  randn!(peps)
  H_local = Models.localham(Models.Model("tfim"), sites; h=1.0)
  losses_gd = gradient_descent(peps, H_local; stepsize=0.005, num_sweeps=num_sweeps)
  losses_ls = optimize(peps, H_local; num_sweeps=num_sweeps, method="GD")
  losses_lbfgs = optimize(peps, H_local; num_sweeps=num_sweeps, method="LBFGS")
  losses_cg = optimize(peps, H_local; num_sweeps=num_sweeps, method="CG")
  for i in 3:(length(losses_gd) - 1)
    @test losses_gd[i] >= losses_gd[i + 1]
    @test losses_ls[i] >= losses_ls[i + 1]
    @test losses_lbfgs[i] >= losses_lbfgs[i + 1]
    @test losses_cg[i] >= losses_cg[i + 1]
  end
end

@testset "test inner product gradient" begin
  Nx = 2
  Ny = 2
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  function loss(peps::PEPS)
    peps_prime = prime(peps; ham=false)
    peps_prime_ham = prime(peps; ham=true)
    network_list = generate_inner_network(peps, peps_prime, peps_prime_ham, [])
    variables = extract_data([peps, peps_prime])
    inners = batch_tensor_contraction(network_list, variables...)
    return sum(inners)[]
  end
  g = gradient(loss, peps)
  inner = inner_network(peps, prime(peps; ham=false))
  g_true_first_site = contract(inner[2:length(inner)])
  g_true_first_site = 2 * g_true_first_site
  @test isapprox(g[1].data[1, 1], g_true_first_site)
end
