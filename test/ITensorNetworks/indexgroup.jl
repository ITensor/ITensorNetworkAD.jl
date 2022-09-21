using ITensors
using ITensorNetworkAD.ITensorNetworks:
  IndexGroup, get_index_groups, get_leaves, neighbor_index_groups
using ITensorNetworkAD.ITensorNetworks:
  inds_network,
  line_network,
  IndexAdjacencyTree,
  find_topo_sort,
  get_ancestors,
  generate_adjacency_tree,
  minswap_adjacency_tree!,
  minswap_adjacency_tree

@testset "test generate_adjacency_tree" begin
  N = (3, 3)
  tn_inds = inds_network(N...; linkdims=2, periodic=false)
  tn = vec(map(inds -> randomITensor(inds...), tn_inds))
  ctree = line_network(tn)
  tn_leaves = get_leaves(ctree)
  ctrees = find_topo_sort(ctree; leaves=tn_leaves)
  ctree_to_igs = Dict{Vector,Vector{IndexGroup}}()
  index_groups = get_index_groups(ctree)
  for c in vcat(tn_leaves, ctrees)
    ctree_to_igs[c] = neighbor_index_groups(c, index_groups)
  end

  adj_tree1 = generate_adjacency_tree(
    tn_leaves[4], get_ancestors(ctrees, tn_leaves[4]), ctree_to_igs
  )
  adj_tree2 = generate_adjacency_tree(
    ctrees[2], get_ancestors(ctrees, ctrees[2]), ctree_to_igs
  )
  for adj_tree in [adj_tree1, adj_tree1]
    @assert length(adj_tree.children) == 3
    @assert adj_tree.fixed_order = true
    c1, c2, c3 = adj_tree.children
    @assert length(c1.children) == 1
    @assert length(c2.children) == 2
    @assert length(c3.children) == 1
  end
end

@testset "test minswap_adjacency_tree!" begin
  i = IndexGroup([Index(2, "i")])
  j = IndexGroup([Index(3, "j")])
  k = IndexGroup([Index(2, "k")])
  l = IndexGroup([Index(4, "l")])
  m = IndexGroup([Index(5, "m")])
  n = IndexGroup([Index(5, "n")])
  I = IndexAdjacencyTree(i)
  J = IndexAdjacencyTree(j)
  K = IndexAdjacencyTree(k)
  L = IndexAdjacencyTree(l)
  M = IndexAdjacencyTree(m)
  N = IndexAdjacencyTree(n)
  JKL = IndexAdjacencyTree([J, K, L], false, false)
  tree = IndexAdjacencyTree([I, JKL, M], false, false)
  tree_copy = copy(tree)
  tree2 = IndexAdjacencyTree([i, k, m, j, l], true, true)
  nswaps = minswap_adjacency_tree!(tree, tree2)
  @assert nswaps == 1
  @assert tree.children == [i, m, k, j, l]
  @assert tree.fixed_direction && tree.fixed_order
  # test minswap_adjacency_tree
  tree3 = IndexAdjacencyTree([i, k, n, m], true, true)
  tree4 = IndexAdjacencyTree([l, n, j], true, true)
  out = minswap_adjacency_tree(tree_copy, tree3, tree4)
  @assert out.children in
    [[i, m, k, j, l], [i, m, k, l, j], [m, i, k, j, l], [m, i, k, l, j]]
end
