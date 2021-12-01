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
  connect_inds = intersect(inds(t1), inds(t2))
  if length(connect_inds) <= 1
    return TreeTensor(t1.tensors..., t2.tensors...)
  end
  network = [t1.tensors..., t2.tensors...]
  if length(noncommoninds(network...)) <= 1
    return TreeTensor(contract(network...))
  end
  # TODO: add caching here
  uncontract_inds = noncommoninds(network...)
  inds_btree = mincut_inds_binary_tree(network, uncontract_inds)
  tree = tree_approximation(network, inds_btree; cutoff=cutoff, maxdim=maxdim)
  return TreeTensor(tree...)
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

function tree_approximation(
  network::Vector{ITensor}, inds_btree::Vector; cutoff=1e-15, maxdim=10000
)
  # inds_dict map each inds_btree node to generated indices
  # tnets_dict map each inds_btree node to a tensor network
  inds_dict = Dict()
  tnets_dict = Dict()
  projectors = []

  function insert_projectors(tree::Vector)
    insert_projectors(tree[1])
    insert_projectors(tree[2])
    # get the tensor network of the current tree node
    net1, net2 = tnets_dict[tree[1]], tnets_dict[tree[2]]
    net_bra = vcat(network, symdiff(net1, net2))
    # get the two indices to merge at this step
    ind1, ind2 = inds_dict[tree[1]], inds_dict[tree[2]]
    # form normal equations to the system to factorize
    net_ket = sim(linkinds, net_bra)
    sim_dict = Dict()
    for ind in [ind1..., ind2...]
      sim_dict[ind] = sim(ind)
    end
    net_ket = sim([ind1..., ind2...], net_ket, sim_dict)
    net_normal = vcat(net_bra, net_ket)
    tensor_normal = contract(generate_optimal_tree(net_normal))
    # use eig to factorize and get the projector
    rinds = (ind1..., ind2...)
    linds = map(i -> sim_dict[i], rinds)
    diag, U = eigen(
      tensor_normal, linds, rinds; cutoff=cutoff, maxdim=maxdim, ishermitian=true
    )
    dr = commonind(diag, U)
    # add the projector to the list projectors
    projectors = vcat(projectors, [U])
    tnets_dict[tree] = vcat(net_bra, [U])
    return inds_dict[tree] = (dr,)
  end

  function insert_projectors(tree::Union{Vector{<:Index},<:Index})
    inds_dict[tree] = Tuple(tree)
    return tnets_dict[tree] = network
  end

  @assert (length(inds_btree) >= 2)
  insert_projectors(inds_btree[1])
  insert_projectors(inds_btree[2])
  last_network = Vector{ITensor}(vcat(network, projectors))
  last_tensor = contract(generate_optimal_tree(last_network))
  return vcat(projectors, [last_tensor])
end
