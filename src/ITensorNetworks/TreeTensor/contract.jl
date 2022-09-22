
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
  inds_btree=nothing;
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
  # # TODO: may want to remove this
  # if inds_groups != nothing
  #   deltainds = vcat(filter(g -> length(g) > 1, inds_groups)...)
  #   deltas, tnprime, _ = split_deltas(deltainds, tn)
  #   tn = Vector{ITensor}(vcat(deltas, tnprime))
  # end
  if inds_btree == nothing
    inds_btree = inds_binary_tree(tn, nothing; algorithm=algorithm)
  end
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
  children::Union{Vector{IndexAdjacencyTree},Vector{IndexGroup}}
  fixed_direction::Bool
  fixed_order::Bool
end

function Base.copy(tree::IndexAdjacencyTree)
  node_to_copynode = Dict{IndexAdjacencyTree,IndexAdjacencyTree}()
  for node in find_topo_sort(tree; type=IndexAdjacencyTree)
    if node.children isa Vector{IndexGroup}
      node_to_copynode[node] = IndexAdjacencyTree(
        node.children, node.fixed_direction, node.fixed_order
      )
      continue
    end
    copynode = IndexAdjacencyTree(
      [node_to_copynode[n] for n in node.children], node.fixed_direction, node.fixed_order
    )
    node_to_copynode[node] = copynode
  end
  return node_to_copynode[tree]
end

function Base.show(io::IO, tree::IndexAdjacencyTree)
  out_str = "\n"
  stack = [tree]
  node_to_level = Dict{IndexAdjacencyTree,Int}()
  node_to_level[tree] = 0
  # pre-order traversal
  while length(stack) != 0
    node = pop!(stack)
    indent_vec = ["  " for _ in 1:node_to_level[node]]
    indent = string(indent_vec...)
    if node.children isa Vector{IndexGroup}
      for c in node.children
        out_str = out_str * indent * string(c) * "\n"
      end
    else
      out_str =
        out_str *
        indent *
        "AdjTree: [fixed_direction]: " *
        string(node.fixed_direction) *
        " [fixed_order]: " *
        string(node.fixed_order) *
        "\n"
      for c in node.children
        node_to_level[c] = node_to_level[node] + 1
        push!(stack, c)
      end
    end
  end
  return print(io, out_str)
end

function IndexAdjacencyTree(index_group::IndexGroup)
  return IndexAdjacencyTree([index_group], false, false)
end

function get_leaves(tree::IndexAdjacencyTree)
  if tree.children isa Vector{IndexGroup}
    return tree.children
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
  if ancestor.children isa Vector{IndexGroup}
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

function reorder_to_right!(
  ancestor::IndexAdjacencyTree, filter_children::Vector{IndexAdjacencyTree}
)
  remain_children = setdiff(ancestor.children, filter_children)
  @assert length(filter_children) >= 1
  @assert length(remain_children) >= 1
  if length(remain_children) == 1
    new_child1 = remain_children[1]
  else
    new_child1 = IndexAdjacencyTree(remain_children, false, false)
  end
  if length(filter_children) == 1
    new_child2 = filter_children[1]
  else
    new_child2 = IndexAdjacencyTree(filter_children, false, false)
  end
  ancestor.children = [new_child1, new_child2]
  return ancestor.fixed_order = true
end

"""
reorder adj_tree based on adj_igs
"""
function reorder!(adj_tree::IndexAdjacencyTree, adj_igs::Set{IndexGroup}; boundary="right")
  @assert boundary in ["left", "right"]
  if boundary_state(adj_tree, adj_igs) == "all"
    return false
  end
  adj_trees = find_topo_sort(adj_tree; type=IndexAdjacencyTree)
  ancestors = [tree for tree in adj_trees if contains(tree, adj_igs)]
  ancestor_to_state = Dict{IndexAdjacencyTree,String}()
  # get the boundary state
  for ancestor in ancestors
    state = boundary_state(ancestor, adj_igs)
    if state == "invalid"
      return false
    end
    ancestor_to_state[ancestor] = state
  end
  # update ancestors
  for ancestor in ancestors
    # reorder
    if ancestor_to_state[ancestor] == "left"
      ancestor.children = reverse(ancestor.children)
    elseif ancestor_to_state[ancestor] == "middle"
      @assert ancestor.fixed_order == false
      filter_children = filter(a -> contains(a, adj_igs), ancestor.children)
      reorder_to_right!(ancestor, filter_children)
    end
    # merge
    if ancestor.fixed_order && ancestor.children isa Vector{IndexAdjacencyTree}
      new_children = Vector{IndexAdjacencyTree}()
      for child in ancestor.children
        if !child.fixed_order
          push!(new_children, child)
        else
          push!(new_children, child.children...)
        end
      end
      ancestor.children = new_children
    end
  end
  # check boundary
  if boundary == "left"
    for ancestor in ancestors
      ancestor.children = reverse(ancestor.children)
    end
  end
  return true
end

"""
Update both keys and values in igs_to_adjacency_tree based on list_adjacent_igs
"""
function update_igs_to_adjacency_tree!(
  list_adjacent_igs::Vector, igs_to_adjacency_tree::Dict{Set{IndexGroup},IndexAdjacencyTree}
)
  function update!(root_igs, adjacent_igs)
    if !haskey(root_igs_to_adjacent_igs, root_igs)
      root_igs_to_adjacent_igs[root_igs] = adjacent_igs
    else
      val = root_igs_to_adjacent_igs[root_igs]
      root_igs_to_adjacent_igs[root_igs] = union(val, adjacent_igs)
    end
  end
  # get each root igs, get the adjacent igs needed. TODO: do we need to consider boundaries here?
  root_igs_to_adjacent_igs = Dict{Set{IndexGroup},Set{IndexGroup}}()
  for adjacent_igs in list_adjacent_igs
    for root_igs in keys(igs_to_adjacency_tree)
      if issubset(adjacent_igs, root_igs)
        update!(root_igs, adjacent_igs)
      end
    end
  end
  if length(root_igs_to_adjacent_igs) == 1
    return nothing
  end
  # if at least 3: for now just put everything together
  if length(root_igs_to_adjacent_igs) >= 3
    root_igs = keys(root_igs_to_adjacent_igs)
    root = union(root_igs...)
    igs_to_adjacency_tree[root] = IndexAdjacencyTree(
      [igs_to_adjacency_tree[r] for r in root_igs], false, false
    )
    for r in root_igs
      delete!(igs_to_adjacency_tree, r)
    end
  end
  # if 2: assign adjacent_igs to boundary of root_igs (if possible), then concatenate
  igs1, igs2 = collect(keys(root_igs_to_adjacent_igs))
  reordered_1 = reorder!(
    igs_to_adjacency_tree[igs1], root_igs_to_adjacent_igs[igs1]; boundary="right"
  )
  reordered_2 = reorder!(
    igs_to_adjacency_tree[igs2], root_igs_to_adjacent_igs[igs2]; boundary="left"
  )
  adj_tree_1 = igs_to_adjacency_tree[igs1]
  adj_tree_2 = igs_to_adjacency_tree[igs2]
  if (!reordered_1) && (!reordered_2)
    out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2], false, false)
  elseif (!reordered_1)
    out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2.children...], false, true)
  elseif (!reordered_2)
    out_adj_tree = IndexAdjacencyTree([adj_tree_1.children..., adj_tree_2], false, true)
  else
    out_adj_tree = IndexAdjacencyTree(
      [adj_tree_1.children..., adj_tree_2.children...], false, true
    )
  end
  root_igs = keys(root_igs_to_adjacent_igs)
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
  igs_to_adjacency_tree = Dict{Set{IndexGroup},IndexAdjacencyTree}()
  for ig in ctree_to_igs[ctree]
    ig_to_adjacent_igs[ig] = Set([ig])
    igs_to_adjacency_tree[Set([ig])] = IndexAdjacencyTree(ig)
  end
  for a in ancestors
    inter_igs = intersect(ctree_to_igs[a[1]], ctree_to_igs[a[2]])
    if issubset(get_leaves(ctree), get_leaves(a[1]))
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

function get_ancestors(ctrees, node)
  return [a for a in ctrees if (issubset(get_leaves(node), get_leaves(a)) && node != a)]
end

"""
Mutates `v` by sorting elements `x[lo:hi]` using the insertion sort algorithm.
This method is a copy-paste-edit of sort! in base/sort.jl, amended to return the bubblesort distance.
"""
function insertion_sort(v::Vector, lo::Int, hi::Int)
  v = copy(v)
  if lo == hi
    return 0
  end
  nswaps = 0
  for i in (lo + 1):hi
    j = i
    x = v[i]
    while j > lo
      if x < v[j - 1]
        nswaps += 1
        v[j] = v[j - 1]
        j -= 1
        continue
      end
      break
    end
    v[j] = x
  end
  return nswaps
end

function insertion_sort(v1::Vector, v2::Vector)
  value_to_index = Dict{Int,Int}()
  for (i, v) in enumerate(v2)
    value_to_index[v] = i
  end
  new_v1 = [value_to_index[v] for v in v1]
  return insertion_sort(new_v1, 1, length(new_v1))
end

function minswap_adjacency_tree!(adj_tree::IndexAdjacencyTree)
  leaves = Vector{IndexGroup}(get_leaves(adj_tree))
  adj_tree.children = leaves
  adj_tree.fixed_order = true
  return adj_tree.fixed_direction = true
end

function minswap_adjacency_tree!(
  adj_tree::IndexAdjacencyTree, input_tree::IndexAdjacencyTree
)
  nodes = input_tree.children
  node_to_int = Dict{IndexGroup,Int}()
  int_to_node = Dict{Int,IndexGroup}()
  index = 1
  for node in nodes
    node_to_int[node] = index
    int_to_node[index] = node
    index += 1
  end
  for node in find_topo_sort(adj_tree; type=IndexAdjacencyTree)
    if node.children isa Vector{IndexGroup}
      continue
    end
    children_tree = [get_leaves(n) for n in node.children]
    children_order = vcat(children_tree...)
    input_int_order = [node_to_int[n] for n in nodes if n in children_order]
    if node.fixed_order
      perms = [children_tree, reverse(children_tree)]
    else
      perms = collect(permutations(children_tree))
    end
    nswaps = []
    for perm in perms
      int_order = [node_to_int[n] for n in vcat(perm...)]
      push!(nswaps, insertion_sort(int_order, input_int_order))
    end
    children_tree = perms[argmin(nswaps)]
    node.children = vcat(children_tree...)
    node.fixed_order = true
    node.fixed_direction = true
  end
  int_order = [node_to_int[n] for n in adj_tree.children]
  return insertion_sort(int_order, 1, length(int_order))
end

function minswap_adjacency_tree(
  adj_tree::IndexAdjacencyTree,
  input_tree1::IndexAdjacencyTree,
  input_tree2::IndexAdjacencyTree,
)
  leaves_1 = get_leaves(input_tree1)
  leaves_2 = get_leaves(input_tree2)
  inter_igs = intersect(leaves_1, leaves_2)
  leaves_1 = [i for i in leaves_1 if !(i in inter_igs)]
  leaves_2 = [i for i in leaves_2 if !(i in inter_igs)]
  input1 = IndexAdjacencyTree([leaves_1..., leaves_2...], true, true)
  input2 = IndexAdjacencyTree([leaves_1..., reverse(leaves_2)...], true, true)
  input3 = IndexAdjacencyTree([reverse(leaves_1)..., leaves_2...], true, true)
  input4 = IndexAdjacencyTree([reverse(leaves_1)..., reverse(leaves_2)...], true, true)
  inputs = [input1, input2, input3, input4]
  adj_tree_copies = [copy(adj_tree) for _ in 1:4]
  nswaps = [minswap_adjacency_tree!(t, i) for (t, i) in zip(adj_tree_copies, inputs)]
  return adj_tree_copies[argmin(nswaps)]
end

@profile function _approximate_contract_pre_process(tn_leaves, ctrees)
  # mapping each contraction tree to its uncontracted index groups
  ctree_to_igs = Dict{Vector,Vector{IndexGroup}}()
  index_groups = get_index_groups(ctrees[end])
  for c in vcat(tn_leaves, ctrees)
    ctree_to_igs[c] = neighbor_index_groups(c, index_groups)
  end
  # mapping each contraction tree to its index adjacency tree
  ctree_to_adj_tree = Dict{Vector,IndexAdjacencyTree}()
  for leaf in tn_leaves
    ancestors = get_ancestors(ctrees, leaf)
    ctree_to_adj_tree[leaf] = generate_adjacency_tree(leaf, ancestors, ctree_to_igs)
    minswap_adjacency_tree!(ctree_to_adj_tree[leaf])
  end
  for c in ctrees
    ancestors = get_ancestors(ctrees, c)
    if ancestors == []
      continue
    end
    adj_tree = generate_adjacency_tree(c, ancestors, ctree_to_igs)
    ctree_to_adj_tree[c] = minswap_adjacency_tree(
      adj_tree, ctree_to_adj_tree[c[1]], ctree_to_adj_tree[c[2]]
    )
  end
  # mapping each index group to the index group tree
  ig_to_ig_tree = Dict{IndexGroup,IndexGroup}()
  for leaf in tn_leaves
    for ig in ctree_to_igs[leaf]
      if !haskey(ig_to_ig_tree, ig)
        inds_tree = inds_binary_tree(leaf, ig.data; algorithm="mincut")
        ig_to_ig_tree[ig] = IndexGroup(inds_tree, true)
      end
    end
  end
  return ctree_to_igs, ctree_to_adj_tree, ig_to_ig_tree
end

# ctree: contraction tree
# tn: vector of tensors representing a tensor network
# adj_tree: index adjacency tree
# ig: index group
# ig_tree: an index group with a tree hierarchy 
function approximate_contract(ctree::Vector; kwargs...)
  tn_leaves = get_leaves(ctree)
  ctrees = find_topo_sort(ctree; leaves=tn_leaves)
  ctree_to_igs, ctree_to_adj_tree, ig_to_ig_tree = _approximate_contract_pre_process(
    tn_leaves, ctrees
  )
  # mapping each contraction tree to a tensor network
  ctree_to_tn = Dict{Vector,Vector{ITensor}}()
  for leaf in tn_leaves
    ctree_to_tn[leaf] = leaf
  end
  for c in ctrees
    tn = vcat(ctree_to_tn[c[1]], ctree_to_tn[c[2]])
    if ctree_to_igs[c] == []
      ctree_to_tn[c] = [optcontract(tn)]
      continue
    end
    ordered_igs = ctree_to_adj_tree[c].children
    inds_btree = line_to_tree([ig_to_ig_tree[ig].data for ig in ordered_igs])
    ctree_to_tn[c] = approximate_contract(tn, inds_btree; kwargs...)
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
