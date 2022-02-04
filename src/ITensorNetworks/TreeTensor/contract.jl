using ..ITensorAutoHOOT
using ..ITensorAutoHOOT: generate_optimal_tree

# contract into one ITensor
ITensors.ITensor(t::TreeTensor; kwargs...) = contract(collect(t.tensors)...; kwargs...)

ITensors.contract(t1::TreeTensor; kwargs...) = t1

function ITensors.contract(t1::TreeTensor, t2::TreeTensor...; kwargs...)
  ts = mapreduce(t -> t.tensors, vcat, t2)
  return contract(t1, TreeTensor(ts...); kwargs...)
end

ITensors.contract(t_list::Vector{TreeTensor}; kwargs...) = contract(t_list...; kwargs...)

function ITensors.contract(t1::TreeTensor, t2::TreeTensor; cutoff, maxdim)
  # print("\n inputs are ", t1, t2, "\n")
  connect_inds = intersect(inds(t1), inds(t2))
  if length(connect_inds) <= 1
    return TreeTensor(t1.tensors..., t2.tensors...)
  end
  network = [t1.tensors..., t2.tensors...]
  if length(noncommoninds(network...)) <= 1
    return TreeTensor(contract(network...))
  end
  uncontract_inds = noncommoninds(network...)
  inds_btree = inds_binary_tree(network, uncontract_inds; algorithm="mincut")
  # TODO: add caching here
  i1 = noncommoninds(network...)
  treeembedding_dict = tree_embedding(network, inds_btree)
  network = vcat(collect(values(treeembedding_dict))...)
  network = Vector{ITensor}(network)
  i2 = noncommoninds(network...)
  @assert (length(i1) == length(i2))
  tree = tree_approximation(network, inds_btree; cutoff=cutoff, maxdim=maxdim)
  out = TreeTensor(tree...)
  # print("output is ", out, "\n")
  return out
end

function uncontract_inds_binary_tree(path::Vector, uncontract_inds::Vector)
  inds1 = uncontract_inds_binary_tree(path[1], uncontract_inds)
  inds2 = uncontract_inds_binary_tree(path[2], uncontract_inds)
  if inds1 == []
    return inds2
  elseif inds2 == []
    return inds1
  else
    return [inds1, inds2]
  end
end

function uncontract_inds_binary_tree(tensor::ITensor, uncontract_inds::Vector)
  return intersect(inds(tensor), uncontract_inds)
end

# # interlaced HOSVD using caching
# function tree_approximation(
#   treeembedding_dict::Dict, inds_btree::Vector; cutoff=1e-15, maxdim=10000
# )
#   env_upper, env_left, env_right = Dict(), Dict(), Dict()
# end

## implements interlaced HOSVD style truncation
function tree_approximation(
  network::Vector{ITensor}, inds_btree::Vector; cutoff=1e-15, maxdim=10000
)
  # inds_dict map each inds_btree node to generated indices
  inds_dict = Dict()
  projectors = []
  function insert_projectors(tree::Vector)
    if length(tree) == 1
      inds_dict[tree] = Tuple(tree)
      return nothing
    end
    insert_projectors(tree[1])
    insert_projectors(tree[2])
    # get the two indices to merge at this step
    ind1, ind2 = inds_dict[tree[1]], inds_dict[tree[2]]
    U, dr = get_projector(network, [ind1..., ind2...]; cutoff=cutoff, maxdim=maxdim)
    # add the projector to the list projectors
    projectors = vcat(projectors, [U])
    network = vcat(network, [U])
    return inds_dict[tree] = (dr,)
  end
  @assert (length(inds_btree) >= 2)
  insert_projectors(inds_btree[1])
  insert_projectors(inds_btree[2])
  last_tensor = contract(generate_optimal_tree(network))
  return vcat(projectors, [last_tensor])
end

## implements non-interlaced HOSVD style truncation
function tree_approximation_non_interlaced(
  network::Vector{ITensor}, inds_btree::Vector; cutoff=1e-15, maxdim=10000
)
  # inds_dict map each inds_btree node to generated indices
  # tnets_dict map each inds_btree node to a tensor network
  inds_dict = Dict()
  tnets_dict = Dict()
  projectors = []
  function insert_projectors(tree::Vector)
    if length(tree) == 1
      inds_dict[tree] = Tuple(tree)
      return tnets_dict[tree] = network
    end
    insert_projectors(tree[1])
    insert_projectors(tree[2])
    # get the tensor network of the current tree node
    net1, net2 = tnets_dict[tree[1]], tnets_dict[tree[2]]
    net_bra = vcat(network, symdiff(net1, net2))
    # get the two indices to merge at this step
    ind1, ind2 = inds_dict[tree[1]], inds_dict[tree[2]]
    U, dr = get_projector(net_bra, [ind1..., ind2...]; cutoff=cutoff, maxdim=maxdim)
    # add the projector to the list projectors
    projectors = vcat(projectors, [U])
    tnets_dict[tree] = vcat(net_bra, [U])
    return inds_dict[tree] = (dr,)
  end
  @assert (length(inds_btree) >= 2)
  insert_projectors(inds_btree[1])
  insert_projectors(inds_btree[2])
  last_network = Vector{ITensor}(vcat(network, projectors))
  last_tensor = contract(generate_optimal_tree(last_network))
  return vcat(projectors, [last_tensor])
end

# get the projector of net_bra with outinds
function get_projector(net_bra, outinds; cutoff, maxdim)
  # form normal equations to the system to factorize
  net_ket = sim(linkinds, net_bra)
  sim_dict = Dict()
  for ind in outinds
    sim_dict[ind] = sim(ind)
  end
  net_ket = sim(outinds, net_ket, sim_dict)
  net_normal = vcat(net_bra, net_ket)
  tensor_normal = contract(generate_optimal_tree(net_normal))
  # use eig to factorize and get the projector
  rinds = Tuple(outinds)
  linds = map(i -> sim_dict[i], rinds)
  diag, U = eigen(
    tensor_normal, linds, rinds; cutoff=cutoff, maxdim=maxdim, ishermitian=true
  )
  dr = commonind(diag, U)
  return U, dr
end
