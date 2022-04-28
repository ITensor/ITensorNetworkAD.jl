## implements interlaced HOSVD style truncation
@profile function tree_approximation(
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
  last_tensor = optcontract(network)
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
  last_tensor = optcontract(last_network)
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
  net_ket = replaceinds(net_ket, sim_dict)
  net_normal = vcat(net_bra, net_ket)
  tensor_normal = optcontract(net_normal)
  # use eig to factorize and get the projector
  rinds = Tuple(outinds)
  linds = map(i -> sim_dict[i], rinds)
  diag, U = eigen(
    tensor_normal, linds, rinds; cutoff=cutoff, maxdim=maxdim, ishermitian=true
  )
  dr = commonind(diag, U)
  return U, dr
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
