using ITensors, ITensorNetworkAD, AutoHOOT
using ITensorNetworkAD.ITensorNetworks:
  PEPS, inner_network, broadcast_add, broadcast_minus, broadcast_mul, broadcast_inner
using ITensorNetworkAD.ITensorAutoHOOT: generate_optimal_tree

@testset "test peps" begin
  Nx = 4
  Ny = 5
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites)

  for ii in 1:(Ny - 1)
    for jj in 1:(Nx - 1)
      inds1 = inds(peps.data[ii, jj])
      inds2 = inds(peps.data[ii, jj + 1])
      inds3 = inds(peps.data[ii + 1, jj])
      inds4 = inds(peps.data[ii + 1, jj + 1])
      @test length(intersect(inds1, inds2)) == 1
      @test length(intersect(inds1, inds3)) == 1
      @test length(intersect(inds1, inds4)) == 0
    end
  end
end

@testset "test inner product" begin
  Nx = 2
  Ny = 3
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites)
  randn!(peps)
  peps_prime = prime(peps; ham=false)
  inner = inner_network(peps, peps_prime)

  opt_inner = generate_optimal_tree(inner)
  out = contract(opt_inner)
  # output is a scalar
  @test size(out) == ()
end

@testset "test inner product with hamiltonian" begin
  Nx = 3
  Ny = 4
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites)
  randn!(peps)

  opsum = OpSum()
  opsum += 0.5, "S+", 1, "S-", 2
  opsum += 0.5, "S-", 1, "S+", 2
  opsum += "Sz", 1, "Sz", 2
  mpo = MPO(opsum, [sites[2, 2], sites[2, 3]])

  peps_prime = prime(peps; ham=false)
  peps_prime_ham = prime(peps; ham=true)
  inner = inner_network(peps, peps_prime, peps_prime_ham, mpo, [2 => 2, 2 => 3])
  opt_inner = generate_optimal_tree(inner)
  out = contract(opt_inner)
  # output is a scalar
  @test size(out) == ()
end

@testset "test plus, minus, multiplication" begin
  Nx = 2
  Ny = 3
  sites = siteinds("S=1/2", Ny, Nx)
  peps1 = PEPS(sites)
  randn!(peps1)
  peps2 = broadcast_mul(1.5, peps1)
  peps3 = broadcast_add(peps1, peps2)
  peps4 = broadcast_minus(peps1, peps2)
  for i in 1:Nx
    for j in 1:Ny
      @test isapprox(peps2.data[j, i], 1.5 * peps1.data[j, i])
      @test isapprox(peps3.data[j, i], peps1.data[j, i] + peps2.data[j, i])
      @test isapprox(peps4.data[j, i], peps1.data[j, i] - peps2.data[j, i])
    end
  end
end