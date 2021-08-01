using ITensors, ITensorNetworkAD, AutoHOOT, Zygote, OptimKit
using ITensorNetworkAD.ITensorNetworks:
  PEPS,
  inner_network,
  Models,
  flatten,
  insert_projectors,
  split_network,
  generate_inner_network
using ITensorNetworkAD.Optimizations: gradient_descent
using ITensorNetworkAD.ITensorAutoHOOT: batch_tensor_contraction

@testset "test monotonic loss decrease of optimization" begin
  Nx, Ny = 2, 3
  num_sweeps = 20
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=10)
  randn!(peps)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  losses_gd = gradient_descent(peps, H_line; stepsize=0.005, num_sweeps=num_sweeps)
  losses_ls = optimize(peps, H_line; num_sweeps=num_sweeps, method="GD")
  losses_lbfgs = optimize(peps, H_line; num_sweeps=num_sweeps, method="LBFGS")
  losses_cg = optimize(peps, H_line; num_sweeps=num_sweeps, method="CG")
  for i in 3:(length(losses_gd) - 1)
    @test losses_gd[i] >= losses_gd[i + 1]
    @test losses_ls[i] >= losses_ls[i + 1]
    @test losses_lbfgs[i] >= losses_lbfgs[i + 1]
    @test losses_cg[i] >= losses_cg[i + 1]
  end
end

@testset "test the equivalence of local and line hamiltonian" begin
  Nx, Ny = 2, 3
  num_sweeps = 3
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=10)
  randn!(peps)
  H_local = Models.localham(Models.Model("tfim"), sites; h=1.0)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  losses_line = gradient_descent(peps, H_line; stepsize=0.005, num_sweeps=num_sweeps)
  losses_local = gradient_descent(peps, H_local; stepsize=0.005, num_sweeps=num_sweeps)
  @test isapprox(losses_local, losses_line)
end

@testset "test monotonic loss decrease of optimization with inserting projectors" begin
  Nx, Ny = 3, 3
  num_sweeps = 20
  cutoff = 1e-15
  maxdim = 100
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  losses_gd = gradient_descent(
    peps,
    H_line,
    insert_projectors;
    stepsize=0.005,
    num_sweeps=num_sweeps,
    cutoff=cutoff,
    maxdim=maxdim,
  )
  losses_ls = optimize(
    peps,
    H_line,
    insert_projectors;
    num_sweeps=num_sweeps,
    method="GD",
    cutoff=cutoff,
    maxdim=maxdim,
  )
  losses_lbfgs = optimize(
    peps,
    H_line,
    insert_projectors;
    num_sweeps=num_sweeps,
    method="LBFGS",
    cutoff=cutoff,
    maxdim=maxdim,
  )
  losses_cg = optimize(
    peps,
    H_line,
    insert_projectors;
    num_sweeps=num_sweeps,
    method="CG",
    cutoff=cutoff,
    maxdim=maxdim,
  )
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
    peps_prime = prime(linkinds, peps)
    peps_prime_ham = prime(peps)
    network_list = generate_inner_network(peps, peps_prime, peps_prime_ham, [])
    variables = flatten([peps, peps_prime])
    inners = batch_tensor_contraction(network_list, variables...)
    return sum(inners)[]
  end
  g = gradient(loss, peps)
  inner = inner_network(peps, prime(linkinds, peps))
  g_true_first_site = contract(inner[2:length(inner)])
  g_true_first_site = 2 * g_true_first_site
  @test isapprox(g[1].data[1, 1], g_true_first_site)
end

@testset "test split network" begin
  Nx, Ny = 3, 3
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  center = (div(size(peps.data)[1] - 1, 2) + 1, :)
  function loss(peps::PEPS)
    tn_split, projectors = insert_projectors(peps, center)
    peps_bra = addtags(linkinds, peps, "bra")
    peps_ket = addtags(linkinds, peps, "ket")
    peps_bra_split = split_network(peps_bra)
    peps_ket_split = split_network(peps_ket)
    network_list = generate_inner_network(
      peps_bra_split, peps_ket_split, peps_ket_split, projectors, []
    )
    variables = flatten([peps_bra_split, peps_ket_split])
    inners = batch_tensor_contraction(network_list, variables...)
    return sum(inners)[]
  end
  g = gradient(loss, peps)
  inner = inner_network(peps, prime(linkinds, peps))
  g_true_first_site = contract(inner[2:length(inner)])
  g_true_first_site = 2 * g_true_first_site
  @test isapprox(g[1].data[1, 1], g_true_first_site)
end

@testset "test inner product gradient with tagging" begin
  Nx, Ny = 3, 3
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  function loss(peps::PEPS)
    peps_bra = addtags(linkinds, peps, "bra")
    peps_ket = addtags(linkinds, peps, "ket")
    sites = commoninds(peps_bra, peps_ket)
    peps_ket_ham = prime(sites, peps_ket)
    projectors = [ITensor(1.0)]
    network_list = generate_inner_network(peps_bra, peps_ket, peps_ket_ham, projectors, [])
    variables = flatten([peps_bra, peps_ket])
    inners = batch_tensor_contraction(network_list, variables...)
    return sum(inners)[]
  end
  g = gradient(loss, peps)
  inner = inner_network(peps, prime(linkinds, peps))
  g_true_first_site = contract(inner[2:length(inner)])
  g_true_first_site = 2 * g_true_first_site
  @test isapprox(g[1].data[1, 1], g_true_first_site)
  @test isapprox(loss(peps), contract(inner)[])
end
