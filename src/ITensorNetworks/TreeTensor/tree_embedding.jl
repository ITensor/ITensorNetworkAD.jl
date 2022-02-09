@profile function tree_embedding(network::Vector{ITensor}, inds_btree::Vector)
  # inds_dict map each inds_btree node to output indices at this node
  # tnets_dict map each inds_btree node to a tensor network
  inds_dict = Dict()
  tnets_dict = Dict()
  function embed(tree::Vector)
    if length(tree) == 1
      ind = tree[1]
      sim_dict = Dict([ind => sim(ind)])
      inds_dict[tree] = Tuple([sim_dict[ind]])
      tnets_dict[tree] = [delta(ind, sim_dict[ind])]
      network = sim([ind], network, sim_dict)
      return nothing
    end
    embed(tree[1])
    embed(tree[2])
    ind1, ind2 = inds_dict[tree[1]], inds_dict[tree[2]]
    network, outinds, tnets_dict[tree[1]], tnets_dict[tree[2]] = insert_deltas(
      network, ind1, ind2, tnets_dict[tree[1]], tnets_dict[tree[2]]
    )
    # use mincut to get the subnetwork
    subnetwork = mincut_subnetwork(network, outinds, noncommoninds(network...))
    network = collect(setdiff(network, subnetwork))
    inds_dict[tree] = Tuple(setdiff(noncommoninds(subnetwork...), outinds))
    return tnets_dict[tree] = subnetwork
  end
  @assert (length(inds_btree) >= 2)
  embed(inds_btree)
  return tnets_dict
end

function insert_deltas(network, ind1, ind2, subnet1, subnet2)
  function update_network(inds, network, subnet)
    sim_dict = Dict([ind => sim(ind) for ind in inds])
    network = vcat(network, [delta(i, sim_dict[i]) for i in inds])
    subnet = sim(inds, subnet, sim_dict)
    return network, subnet, collect(values(sim_dict))
  end
  intersect_inds = intersect(ind1, ind2)
  ind1_unique = collect(setdiff(ind1, intersect_inds))
  ind2_unique = collect(setdiff(ind2, intersect_inds))
  outinds = []
  # look at intersect_inds
  if length(intersect_inds) >= 1
    network, subnet1, siminds = update_network(intersect_inds, network, subnet1)
    outinds = vcat(outinds, intersect_inds, siminds)
  end
  # go over ids in t1 but not t2, and t2 but not t1
  if length(ind1_unique) >= 1
    network, subnet1, siminds = update_network(ind1_unique, network, subnet1)
    outinds = vcat(outinds, siminds)
  end
  if length(ind2_unique) >= 1
    network, subnet2, siminds = update_network(ind2_unique, network, subnet2)
    outinds = vcat(outinds, siminds)
  end
  return network, outinds, subnet1, subnet2
end
