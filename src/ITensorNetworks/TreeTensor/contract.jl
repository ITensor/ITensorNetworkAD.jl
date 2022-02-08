using ..ITensorAutoHOOT
using ..ITensorAutoHOOT: generate_optimal_tree

optcontract(t_list::Vector{ITensor}) = contract(generate_optimal_tree(t_list))

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
  # t1 = time()
  # tree_approximation(network, inds_btree; cutoff=cutoff, maxdim=maxdim)
  # t2 = time()
  # print("tree approximation without caching runs ", t2 - t1, "s\n")
  i1 = noncommoninds(network...)
  embedding = tree_embedding(network, inds_btree)
  network = Vector{ITensor}(vcat(collect(values(embedding))...))
  i2 = noncommoninds(network...)
  @assert (length(i1) == length(i2))
  t1 = time()
  tree = tree_approximation(embedding, inds_btree; cutoff=cutoff, maxdim=maxdim)
  t2 = time()
  # print("tree approximation with caching runs ", t2 - t1, "s\n")
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

# interlaced HOSVD using caching
function tree_approximation(embedding::Dict, inds_btree::Vector; cutoff=1e-15, maxdim=10000)
  projectors = []
  # initialize sim_dict
  network = vcat(collect(values(embedding))...)
  uncontractinds = noncommoninds(network...)
  innerinds = mapreduce(t -> [i for i in inds(t)], vcat, network)
  innerinds = Vector(setdiff(innerinds, uncontractinds))
  siminner_dict = Dict([ind => sim(ind) for ind in innerinds])
  function closednet(tree)
    netbra = embedding[tree]
    netket = sim(innerinds, netbra, siminner_dict)
    if length(tree) == 1
      return optcontract(vcat(netbra, netket))
    end
    tleft, tright = closednet(tree[1]), closednet(tree[2])
    return optcontract(vcat(netbra, netket, [tleft], [tright]))
  end
  function insert_projectors(tree::Vector, env::ITensor)
    netbra = embedding[tree]
    netket = sim(innerinds, netbra, siminner_dict)
    if length(tree) == 1
      tensor_bra = optcontract(netbra)
      tensor_ket = sim(innerinds, [tensor_bra], siminner_dict)[1]
      inds_pair = (tree[1], sim(tree[1]))
      tensor_ket = sim(tree, [tensor_ket], Dict([inds_pair[1] => inds_pair[2]]))[1]
      return inds_pair, optcontract([netbra..., netket...]), [tensor_bra, tensor_ket]
    end
    # update children
    envnet = [env, closednet(tree[2]), netbra..., netket...]
    ind1_pair, subnetsq1, subnet1 = insert_projectors(tree[1], optcontract(envnet))
    envnet = [env, subnetsq1, netbra..., netket...]
    ind2_pair, _, subnet2 = insert_projectors(tree[2], optcontract(envnet))
    # compute the projector
    rinds = (ind1_pair[1], ind2_pair[1])
    linds = (ind1_pair[2], ind2_pair[2])
    net = [env, netbra..., netket..., subnet1..., subnet2...]
    tnormal = optcontract(net)
    diag, U = eigen(tnormal, linds, rinds; cutoff=cutoff, maxdim=maxdim, ishermitian=true)
    dr = commonind(diag, U)
    Usim = replaceinds(U, rinds => linds)
    net = [netbra..., netket..., subnet1..., subnet2..., U, Usim]
    subnetsq = optcontract(net)
    net1 = [netbra..., subnet1[1], subnet2[1], U]
    dr_pair = (dr, sim(dr))
    Usim = replaceinds(Usim, [dr_pair[1]] => [dr_pair[2]])
    net2 = [netket..., subnet1[2], subnet2[2], Usim]
    subnet = [optcontract(net1), optcontract(net2)]
    # add the projector to the list projectors
    projectors = vcat(projectors, [U])
    return dr_pair, subnetsq, subnet
  end
  @assert (length(inds_btree) >= 2)
  bra = embedding[inds_btree]
  ket = sim(innerinds, bra, siminner_dict)
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
  net_ket = sim(outinds, net_ket, sim_dict)
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
