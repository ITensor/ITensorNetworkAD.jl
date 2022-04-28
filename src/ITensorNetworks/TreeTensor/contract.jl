
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

function approximate_contract(tn::Vector, inds_groups=nothing; kwargs...)
  if inds_groups == nothing
    inds_groups = noncommoninds(vectorize(tn)...)
  end
  @assert length(tn) == 2
  left_inds = uncontractinds(tn[1])
  right_inds = uncontractinds(tn[2])
  inter_inds = intersect(left_inds, right_inds)
  # form inds_groups of tn[1] and tn[2]
  inds_groups_left = subtree(inds_groups, left_inds)
  inds_groups_right = subtree(inds_groups, right_inds)
  inds_groups_left = merge_tree(inds_groups_left, inter_inds; append=true)
  inds_groups_right = merge_tree(inds_groups_right, inter_inds; append=true)
  @assert length(vectorize(inds_groups_left)) == length(left_inds) &&
    length(vectorize(inds_groups_right)) == length(right_inds)
  out_left = approximate_contract(tn[1], inds_groups_left; kwargs...)
  out_right = approximate_contract(tn[2], inds_groups_right; kwargs...)
  return approximate_contract([out_left..., out_right...], inds_groups; kwargs...)
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
