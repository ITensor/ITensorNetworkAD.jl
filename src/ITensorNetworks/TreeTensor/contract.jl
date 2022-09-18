
ITensors.enable_contraction_sequence_optimization()

@profile myeigen(tnormal, linds, rinds; kwargs...) = eigen(tnormal, linds, rinds; kwargs...)

@profile function optcontract(t_list::Vector)
  if length(t_list) == 0
    return ITensor(1.0)
  end
  # for t in t_list
  #   @info "size of t is", size(t)
  # end
  return contract(generate_optimal_tree(t_list))
end

# contract into one ITensor
ITensors.ITensor(t::TreeTensor; kwargs...) = contract(collect(t.tensors)...; kwargs...)

function ITensors.contract(t1::TreeTensor, t2::TreeTensor...; kwargs...)
  ts = mapreduce(t -> t.tensors, vcat, t2)
  return contract(t1, TreeTensor(ts...); kwargs...)
end

# TODO: call approximate_contract
ITensors.contract(t_list::Vector{TreeTensor}; kwargs...) = contract(t_list...; kwargs...)

function ITensors.contract(t1::TreeTensor, t2::TreeTensor; kwargs...)
  # @info "inputs are $(t1), $(t2)"
  connect_inds = intersect(inds(t1), inds(t2))
  if length(connect_inds) <= 1
    return TreeTensor(t1.tensors..., t2.tensors...)
  end
  network = [t1.tensors..., t2.tensors...]
  return contract(TreeTensor(network...); kwargs...)
end

function ITensors.contract(t::TreeTensor; kwargs...)
  network = t.tensors
  out = approximate_contract(network; kwargs...)
  out = TreeTensor(out)
  # @info "output is $(out)"
  return out
end

approximate_contract(tn::ITensor, inds_groups; kwargs...) = [tn]

function approximate_contract(
  tn::Vector{ITensor},
  inds_groups=nothing;
  cutoff,
  maxdim,
  maxsize=10^15,
  algorithm="mincut-mps",
)
  uncontract_inds = noncommoninds(tn...)
  allinds = collect(Set(mapreduce(t -> collect(inds(t)), vcat, tn)))
  innerinds = setdiff(allinds, uncontract_inds)
  if length(uncontract_inds) <= 2
    return [optcontract(tn)]
  end
  # cases where tn is a tree, or contains 2 disconnected trees
  if length(innerinds) <= length(tn) - 1
    return tn
  end
  # TODO: may want to remove this
  if inds_groups != nothing
    deltainds = vcat(filter(g -> length(g) > 1, inds_groups)...)
    deltas, tnprime, _ = split_deltas(deltainds, tn)
    tn = Vector{ITensor}(vcat(deltas, tnprime))
  end
  inds_btree = inds_binary_tree(tn, inds_groups; algorithm=algorithm)
  # tree_approximation(tn, inds_btree; cutoff=cutoff, maxdim=maxdim)
  embedding = tree_embedding(tn, inds_btree)
  tn = Vector{ITensor}(vcat(collect(values(embedding))...))
  i2 = noncommoninds(tn...)
  @assert (length(uncontract_inds) == length(i2))
  tree = tree_approximation_cache(
    embedding, inds_btree; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize
  )
  return tree
end

function uncontractinds(tn)
  if tn isa ITensor
    return inds(tn)
  else
    return noncommoninds(vectorize(tn)...)
  end
end

# Note that the children ordering matters here.
mutable struct IndexAdjacencyTree
  children::Union{Vector{IndexAdjacencyTree}, IndexGroup}
  fixed_direction::Bool
  fixed_order::Bool
end

function IndexAdjacencyTree(index_group::IndexGroup)
  return IndexAdjacencyTree(index_group, false, false)
end

function get_leaves(tree::IndexAdjacencyTree)
  if tree.children isa IndexGroup
    return [tree.children]
  end
  leaves = [get_leaves(c) for c in tree.children]
  return vcat(leaves...)
end

function Base.contains(adj_tree::IndexAdjacencyTree, adj_igs::Set{IndexGroup})
  leaves = Set(get_leaves(adj_tree))
  return issubset(adj_igs, leaves)
end

function Base.iterate(x::IndexAdjacencyTree)
  return iterate(x, 1)
end

function Base.iterate(x::IndexAdjacencyTree, index)
  if index > length(x.children)
    return nothing
  end
  return x.children[index], index + 1
end

function boundary_state(ancestor::IndexAdjacencyTree, adj_igs::Set{IndexGroup})
  if ancestor.children isa IndexGroup
    return "all"
  end
  if !ancestor.fixed_order
    filter_children = filter(a -> contains(a, adj_igs), ancestor.children)
    @assert length(filter_children) <= 1 
    if length(filter_children) == 1
      return "middle"
    elseif Set(get_leaves(ancestor)) == adj_igs
      return "all"
    else
      return "invalid"
    end 
  end
  @assert length(ancestor.children) >= 2
  if contains(ancestor.children[1], adj_igs)
    return "left"
  elseif contains(ancestor.children[end], adj_igs)
    return "right"
  elseif Set(get_leaves(ancestor)) == adj_igs
    return "all"
  else
    return "invalid"
  end
end

"""
reorder adj_tree based on adj_igs
"""
function reorder!(adj_tree::IndexAdjacencyTree, adj_igs::Set{IndexGroup}; boundary="right")
  @assert boundary in ["left", "right"]
  adj_trees = find_topo_sort(adj_tree)
  ancestors = [tree for tree in adj_trees if contains(tree, adj_igs)]
  ancester_to_state = Dict{IndexAdjacencyTree, String}()
  # get the boundary state
  for ancestor in ancestors
    state = boundary_state(ancestor, adj_igs)
    if state == "invalid"
      return false
    end
    ancester_to_state[ancestor] = state
  end
  # update ancestors
  for ancestor in ancestors
    if ancester_to_state[ancestor] == "left"
      ancestor.children = reverse(ancestor.children)
    elseif ancester_to_state[ancestor] == "middle"
      @assert ancestor.fixed_order == false
      filter_children = filter(a -> contains(a, adj_igs), ancestor.children)
      new_child1 = IndexAdjacencyTree(setdiff(ancestor.children, filter_children), false, false)
      new_child2 = IndexAdjacencyTree(filter_children, false, false)
      ancestor.children = [new_child1, new_child2]
      ancestor.fixed_order = true
    end
  end
  # check boundary
  if boundary == "left"
    for ancestor in ancestors
      ancestor.children = reverse(ancestor.children)
    end
  end
end

"""
Update both keys and values in igs_to_adjacency_tree based on list_adjacent_igs
"""
function update_igs_to_adjacency_tree!(list_adjacent_igs::Vector, igs_to_adjacency_tree::Dict{Set{IndexGroup}, IndexAdjacencyTree})
  function update!(root_igs, adjacent_igs)
    if !haskey(root_igs_to_adjacent_igs, root_igs)
      root_igs_to_adjacent_igs[root_igs] = adjacent_igs
    else
      val = root_igs_to_adjacent_igs[root_igs]
      root_igs_to_adjacent_igs[root_igs] = union(val, adjacent_igs)
    end
  end
  # get each root igs, get the adjacent igs needed. TODO: do we need to consider boundaries here?
  root_igs_to_adjacent_igs = Dict{Set{IndexGroup}, Set{IndexGroup}}()
  for adjacent_igs in list_adjacent_igs
    for root_igs in keys(igs_to_adjacency_tree)
      if issubset(adjacent_igs, root_igs)
        update!(root_igs, adjacent_igs)
      end
    end
  end
  if length(root_igs_to_adjacent_igs) == 1
    return
  end
  # if at least 3: for now just put everything together
  if length(root_igs_to_adjacent_igs) >= 3
    root_igs = keys(root_igs_to_adjacent_igs)
    root = union(root_igs...)
    igs_to_adjacency_tree[root] = IndexAdjacencyTree([igs_to_adjacency_tree[r] for r in root_igs], false, false)
    for r in root_igs
      delete!(igs_to_adjacency_tree, r)
    end
  end
  # if 2: assign adjacent_igs to boundary of root_igs (if possible), then concatenate
  igs1, igs2 = collect(keys(root_igs_to_adjacent_igs))
  reordered_1 = reorder!(igs_to_adjacency_tree[igs1], root_igs_to_adjacent_igs[igs1]; boundary="right")
  reordered_2 = reorder!(igs_to_adjacency_tree[igs2], root_igs_to_adjacent_igs[igs2]; boundary="left")
  adj_tree_1 = igs_to_adjacency_tree[igs1]
  adj_tree_2 = igs_to_adjacency_tree[igs2]
  if (!reordered_1) && (!reordered_2)
    out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2], false, false)
  elseif (!reordered_1)
    out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2.children...], false, true)
  elseif (!reordered_2)
    out_adj_tree = IndexAdjacencyTree([adj_tree_1.children..., adj_tree_2], false, true)
  else
    out_adj_tree = IndexAdjacencyTree([adj_tree_1.children..., adj_tree_2.children...], false, true)
  end
  root = union(root_igs...)
  igs_to_adjacency_tree[root] = out_adj_tree
  for r in root_igs
    delete!(igs_to_adjacency_tree, r)
  end
end

"""
Generate the adjacency tree of a contraction tree
Args:
==========
ctree: the input contraction tree
ancestors: ancestor ctrees of the input ctree
ctree_to_igs: mapping each ctree to neighboring index groups 
"""
function generate_adjacency_tree(ctree, ancestors, ctree_to_igs)
  # mapping each index group to adjacent input igs
  ig_to_adjacent_igs = Dict{IndexGroup,Set{IndexGroup}}()
  # mapping each igs to an adjacency tree
  igs_to_adjacency_tree = Dict{Set{IndexGroup}, IndexAdjacencyTree}()
  for ig in ctree_to_igs[ctree]
    ig_to_adjacent_igs[ig] = Set(ig)
    igs_to_adjacency_tree[Set(ig)] = IndexAdjacencyTree(ig)
  end
  for a in ancestors
    inter_igs = intersect(ctree_to_igs[a[1]], ctree_to_igs[a[2]])
    if ctree in a[1]
      new_igs = setdiff(ctree_to_igs[a[2]], inter_igs)
    else
      new_igs = setdiff(ctree_to_igs[a[1]], inter_igs)
    end
    # Tensor product is not considered for now
    @assert length(inter_igs) >= 1
    list_adjacent_igs = [ig_to_adjacent_igs[ig] for ig in inter_igs]
    update_igs_to_adjacency_tree!(list_adjacent_igs, igs_to_adjacency_tree)
    for ig in new_igs
      ig_to_adjacent_igs[ig] = union(list_adjacent_igs...)
    end
    if length(igs_to_adjacency_tree) == 1
      return collect(values(igs_to_adjacency_tree))[1]
    end
  end
end

# ctree: contraction tree
# tn: vector of tensors representing a tensor network
# adj_tree: index adjacency tree
# ig: index group
# ig_tree: an index group with a tree hierarchy 
function approximate_contract(ctree::Vector; kwargs...)
  index_groups = get_index_groups(ctree)
  tn_leaves = get_leaves(ctree)
  ctrees = find_topo_sort(ctree, tn_leaves)
  # mapping each contraction tree to its uncontracted index groups
  ctree_to_igs = Dict{Vector,Vector{IndexGroup}}()
  for c in vcat(tn_leaves, ctrees)
    ctree_to_igs[c] = neighbor_index_groups(c, index_groups)
  end
  # mapping each contraction tree to its index adjacency tree
  ctree_to_adj_tree = Dict{Vector,IndexAdjacencyTree}()
  for leaf in tn_leaves
    ancestors = [a for a in ctrees if (leaf in a && leaf != a)]
    # TODO: implement this
    ctree_to_adj_tree[leaf] = generate_adjacency_tree(
      leaf,
      ancestors,
      ctree_to_igs
    )
  end
  # mapping each contraction tree to a tensor network
  ctree_to_tn = Dict{Vector,Vector{ITensor}}()
  for leaf in tn_leaves
    ctree_to_tn[leaf] = leaf
  end
  # mapping each index group to the index group tree
  ig_to_ig_tree = Dict{IndexGroup,IndexGroup}()
  for leaf in tn_leaves
    for ig in ctree_to_igs[leaf]
      if !haskey(ig_to_ig_tree, ig)
        # TODO: implement this
        ig_to_ig_tree[ig] = construct_index_group_tree(ig, leaf)
      end 
    end 
  end
  for c in ctrees
    ancestors = [a for a in ctrees if (c in a && c != a)]
    adj_tree = generate_adjacency_tree(
      c,
      ancestors,
      ctree_to_igs
    )
    # TODO: get the input line ordering
    ctree_to_adj_tree[c] = minswap_adjacency_tree(
      adj_tree,
      ctree_to_adj_tree[c[1]],
      ctree_to_adj_tree[c[2]]
    )
    # TODO: implement this
    ctree_to_tn[c] = approximate_contract(
      vcat(ctree_to_tn[c[1]], ctree_to_tn[c[2]]),
      ctree_to_adj_tree[c],
      ig_to_ig_tree;
      kwargs...,
    )
  end
  return ctree_to_tn[ctrees[end]]
end

# interlaced HOSVD using caching
@profile function tree_approximation_cache(
  embedding::Dict, inds_btree::Vector; cutoff=1e-15, maxdim=10000, maxsize=10000
)
  projectors = []
  # initialize sim_dict
  network = vcat(collect(values(embedding))...)
  uncontractinds = noncommoninds(network...)
  innerinds = mapreduce(t -> [i for i in inds(t)], vcat, network)
  innerinds = Vector(setdiff(innerinds, uncontractinds))
  siminner_dict = Dict([ind => sim(ind) for ind in innerinds])

  function closednet(tree)
    netbra = embedding[tree]
    netket = replaceinds(netbra, siminner_dict)
    if length(tree) == 1
      return optcontract(vcat(netbra, netket))
    end
    tleft, tright = closednet(tree[1]), closednet(tree[2])
    return optcontract(vcat(netbra, netket, [tleft], [tright]))
  end

  function insert_projectors(tree::Vector, env::ITensor)
    netbra = embedding[tree]
    netket = replaceinds(netbra, siminner_dict)
    if length(tree) == 1
      tensor_bra = optcontract(netbra)
      tensor_ket = replaceinds([tensor_bra], siminner_dict)[1]
      inds_pair = (tree[1], sim(tree[1]))
      tensor_ket = replaceinds([tensor_ket], Dict([inds_pair[1] => inds_pair[2]]))[1]
      return inds_pair, optcontract([netbra..., netket...]), [tensor_bra, tensor_ket]
    end
    # update children
    subenvtensor = optcontract([env, netbra...])
    envnet = [subenvtensor, closednet(tree[2]), netket...]
    ind1_pair, subnetsq1, subnet1 = insert_projectors(tree[1], optcontract(envnet))
    envnet = [subenvtensor, subnetsq1, netket...]
    ind2_pair, _, subnet2 = insert_projectors(tree[2], optcontract(envnet))
    # compute the projector
    rinds = (ind1_pair[1], ind2_pair[1])
    linds = (ind1_pair[2], ind2_pair[2])
    # to handle the corner cases where subnet1/subnet2 could be empty
    netket = replaceinds(
      netket, Dict([ind1_pair[1] => ind1_pair[2], ind2_pair[1] => ind2_pair[2]])
    )
    net = [subenvtensor, netket..., subnet1..., subnet2...]
    tnormal = optcontract(net)
    dim2 = floor(maxsize / (space(ind1_pair[1]) * space(ind2_pair[1])))
    dim = min(maxdim, dim2)
    diag, U = myeigen(tnormal, linds, rinds; cutoff=cutoff, maxdim=dim, ishermitian=true)
    dr = commonind(diag, U)
    Usim = replaceinds(U, rinds => linds)
    net1 = [netbra..., subnet1[1], subnet2[1], U]
    net2 = [netket..., subnet1[2], subnet2[2], Usim]
    tensor1 = optcontract(net1)
    tensor2 = replaceinds(tensor1, noncommoninds(net1...) => noncommoninds(net2...))
    subnetsq = optcontract([tensor1, tensor2])
    dr_pair = (dr, sim(dr))
    tensor2 = replaceinds(tensor2, [dr_pair[1]] => [dr_pair[2]])
    subnet = [tensor1, tensor2]
    # add the projector to the list projectors
    projectors = vcat(projectors, [U])
    return dr_pair, subnetsq, subnet
  end

  @assert (length(inds_btree) >= 2)
  bra = embedding[inds_btree]
  ket = replaceinds(bra, siminner_dict)
  # update children
  envnet = [closednet(inds_btree[2]), bra..., ket...]
  _, netsq1, n1 = insert_projectors(inds_btree[1], optcontract(envnet))
  envnet = [netsq1, bra..., ket...]
  _, _, n2 = insert_projectors(inds_btree[2], optcontract(envnet))
  # last tensor
  envnet = [n1[1], n2[1], bra...]
  last_tensor = optcontract(envnet)
  return vcat(projectors, [last_tensor])
end
