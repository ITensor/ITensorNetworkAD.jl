using ITensors, ITensorNetworkAD, AutoHOOT
using ITensorNetworkAD.ITensorNetworks:
  PEPS,
  Models,
  inner_network,
  broadcast_add,
  broadcast_minus,
  broadcast_mul,
  broadcast_inner,
  insert_projectors,
  split_network,
  inner_networks,
  tree_w_projectors,
  get_leaf_nodes
using ITensorNetworkAD.ITensorAutoHOOT:
  generate_optimal_tree, batch_tensor_contraction, Executor

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
  peps_prime = prime(linkinds, peps)
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

  peps_prime = prime(linkinds, peps)
  peps_prime_ham = prime(peps)
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

@testset "test insert projectors" begin
  Nx, Ny = 3, 3
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  tn_split_row, tn_split_column, projectors_row, projectors_column = insert_projectors(peps)
  for i in 2:length(tn_split_row)
    for (t1, t2) in zip(tn_split_row[1], tn_split_row[i])
      @test t1 == t2
    end
  end
  for i in 2:length(tn_split_column)
    for (t1, t2) in zip(tn_split_column[1], tn_split_column[i])
      @test t1 == t2
    end
  end
end

@testset "test split network with hamiltonian" begin
  Nx, Ny = 3, 3
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)

  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  H_row = [H for H in H_line if H.coord[2] isa Colon]
  H_column = [H for H in H_line if H.coord[1] isa Colon]

  _, _, projectors_row, projectors_column = insert_projectors(peps)
  peps_bra = addtags(linkinds, peps, "bra")
  peps_ket = addtags(linkinds, peps, "ket")
  peps_bra_rot = addtags(linkinds, peps, "brarot")
  peps_ket_rot = addtags(linkinds, peps, "ketrot")
  sites = commoninds(peps_bra, peps_ket)
  peps_bra_split = split_network(peps_bra)
  peps_ket_split = split_network(peps_ket)
  peps_ket_split_ham = prime(sites, peps_ket_split)
  peps_bra_split_rot = split_network(peps_bra_rot, true)
  peps_ket_split_rot = split_network(peps_ket_rot, true)
  peps_ket_split_rot_ham = prime(sites, peps_ket_split_rot)

  for i in 1:length(projectors_row)
    network_list = inner_networks(
      peps_bra_split, peps_ket_split, peps_ket_split_ham, projectors_row[i], [H_row[i]]
    )
    inners = batch_tensor_contraction(network_list)
    @test size(inners[1]) == ()
  end
  for i in 1:length(projectors_column)
    network_list = inner_networks(
      peps_bra_split_rot,
      peps_ket_split_rot,
      peps_ket_split_rot_ham,
      projectors_column[i],
      [H_column[i]],
    )
    inners = batch_tensor_contraction(network_list)
    @test size(inners[1]) == ()
  end
end

@testset "test converting peps based network into a tree" begin
  # ITensors.set_warn_order(9)
  Nx, Ny = 6, 5
  sites = siteinds("S=1/2", Ny, Nx)
  peps = PEPS(sites; linkdims=2)
  randn!(peps)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  H_row = [H for H in H_line if H.coord[2] isa Colon]
  H_column = [H for H in H_line if H.coord[1] isa Colon]
  _, _, projectors_row, projectors_column = insert_projectors(peps)

  peps_bra = addtags(linkinds, peps, "bra")
  peps_ket = addtags(linkinds, peps, "ket")
  peps_bra_rot = addtags(linkinds, peps, "brarot")
  peps_ket_rot = addtags(linkinds, peps, "ketrot")
  sites = commoninds(peps_bra, peps_ket)
  peps_bra_split = split_network(peps_bra)
  peps_ket_split = split_network(peps_ket)
  peps_ket_split_ham = prime(sites, peps_ket_split)
  peps_bra_split_rot = split_network(peps_bra_rot, true)
  peps_ket_split_rot = split_network(peps_ket_rot, true)
  peps_ket_split_rot_ham = prime(sites, peps_ket_split_rot)

  projectors = projectors_row[1]
  peps_tree = inner_network(peps_bra_split, peps_ket_split, projectors, tree_w_projectors)
  leaves = get_leaf_nodes(peps_tree)
  @test length(leaves) == 2 * Nx * Ny + length(projectors)
  executor = Executor([peps_tree])
  out = batch_tensor_contraction(executor)
  @test size(out[1]) == ()
  for i in 1:length(projectors_row)
    peps_tree = inner_networks(
      peps_bra_split,
      peps_ket_split,
      peps_ket_split_ham,
      [projectors_row[i]],
      [H_row[i]],
      tree_w_projectors,
    )[1]
    leaves = get_leaf_nodes(peps_tree)
    @test length(leaves) == 2 * Nx * Ny + length(projectors_row[i]) + Nx
    executor = Executor([peps_tree])
    out = batch_tensor_contraction(executor)
    @test size(out[1]) == ()
  end
  for i in 1:length(projectors_column)
    peps_tree = inner_networks(
      peps_bra_split_rot,
      peps_ket_split_rot,
      peps_ket_split_rot_ham,
      [projectors_column[i]],
      [H_column[i]],
      tree_w_projectors,
    )[1]
    leaves = get_leaf_nodes(peps_tree)
    @test length(leaves) == 2 * Nx * Ny + length(projectors_column[i]) + Ny
    executor = Executor([peps_tree])
    out = batch_tensor_contraction(executor)
    @test size(out[1]) == ()
  end
end
