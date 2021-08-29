using ITensors, ITensorNetworkAD, AutoHOOT, Zygote, OptimKit
using ITensorNetworkAD.ITensorNetworks:
  PEPS,
  inner_network,
  inner_networks,
  Models,
  flatten,
  insert_projectors,
  split_network,
  MPSTensor,
  broadcast_inner
using ITensorNetworkAD.Optimizations:
  gradient_descent, backtracking_linesearch, loss_grad_wrap
using ITensorNetworkAD.ITensorAutoHOOT: batch_tensor_contraction, abstract_network

@testset "test MPSTensor" begin
  Nx, Ny = 3, 3
  num_sweeps = 20
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  H = Models.mpo(Models.Model("tfim"), sites; h=1.0)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  params = Dict(:maxdim => 1000, :cutoff => 1e-15, :method => "general_mps")
  loss_w_grad_mps = loss_grad_wrap(peps, H_line, abstract_network, MPSTensor; params...)
  loss_w_grad = loss_grad_wrap(peps, H_line)
  loss_mps, grad_mps = loss_w_grad_mps(peps)
  loss, grad = loss_w_grad(peps)
  print(loss_mps, loss)
  print(grad_mps, grad)
  g_mps_nrm = broadcast_inner(grad_mps, grad_mps)
  g_nrm = broadcast_inner(grad, grad)
  @test isapprox(loss, loss_mps)
  @test isapprox(g_nrm, g_mps_nrm)
end

@testset "test the loss of optimization" begin
  Nx, Ny = 2, 3
  num_sweeps = 20
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=10)
  randn!(peps)
  psi0 = randomMPS(vec(sites); linkdims=10)
  sweeps = Sweeps(10)
  maxdim!(sweeps, 10)

  H = Models.mpo(Models.Model("tfim"), sites; h=1.0)
  energy, _ = dmrg(H, psi0, sweeps)
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
  @test abs(energy - losses_gd[end]) < 0.5
  @test abs(energy - losses_ls[end]) < 0.5
  @test abs(energy - losses_lbfgs[end]) < 0.5
  @test abs(energy - losses_cg[end]) < 0.5
end

@testset "test the loss of optimization" begin
  Nx, Ny = 3, 3
  num_sweeps = 20
  cutoff = 1e-15
  maxdim = 100
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)

  psi0 = randomMPS(vec(sites); linkdims=3)
  sweeps = Sweeps(10)
  maxdim!(sweeps, 10)

  H = Models.mpo(Models.Model("tfim"), sites; h=1.0)
  energy, _ = dmrg(H, psi0, sweeps)
  print(energy)

  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  losses_gdls = gradient_descent(
    peps,
    H_line,
    insert_projectors,
    backtracking_linesearch;
    beta=0.5,
    num_sweeps=num_sweeps,
    cutoff=cutoff,
    maxdim=maxdim,
  )
  for i in 3:(length(losses_gdls) - 1)
    @test losses_gdls[i] >= losses_gdls[i + 1]
  end
  @test abs(energy - losses_gdls[end]) < 0.5
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
    network_list = [inner_network(peps, peps_prime)]
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
  ITensors.set_warn_order(40)
  Ny, Nx = 3, 4
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  H_line = [H for H in H_line if H.coord[2] isa Colon]
  tn_split_row, tn_split_column, projectors_row, projectors_column = insert_projectors(
    peps, 1e-15, 1000
  )
  for (i, H) in enumerate(H_line)
    function loss(peps::PEPS)
      peps_bra = addtags(linkinds, peps, "bra")
      peps_ket = addtags(linkinds, peps, "ket")
      sites = commoninds(peps_bra, peps_ket)
      peps_bra_split = split_network(peps_bra)
      peps_ket_split = split_network(peps_ket)
      peps_ket_split_ham = prime(sites, peps_ket_split)
      network_list = inner_networks(
        peps_bra_split, peps_ket_split, peps_ket_split_ham, projectors_row[i], [H]
      )
      variables = flatten([peps_bra_split])
      inners = batch_tensor_contraction(network_list, variables...)
      return sum(inners)[]
    end
    g = gradient(loss, peps)
    inner = inner_networks(peps, prime(linkinds, peps), prime(peps), [H])[1]
    g_true_first_site = contract(vcat(inner[2:length(inner)]))
    @test isapprox(g[1].data[1, 1], g_true_first_site)
  end
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
    projectors = [ITensor(1.0)]
    network_list = [inner_network(peps_bra, peps_ket, projectors)]
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
