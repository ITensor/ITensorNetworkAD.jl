mutable struct IndexGroup
  data::Vector
  istree::Bool
end

# TODO: general tags are not comparable
Base.isless(a::Index, b::Index) = id(a) < id(b) || (id(a) == id(b) && plev(a) < plev(b)) # && tags(a) < tags(b)

function IndexGroup(indices::Vector{<:Index})
  return IndexGroup(sort(indices), false)
end

function get_index_groups(tn::Vector)
  leaves = get_leaves(tn)
  igs = []
  for (t1, t2) in powerset(leaves, 2, 2)
    inds = intersect(noncommoninds(t1...), noncommoninds(t2...))
    if length(inds) >= 1
      push!(igs, IndexGroup(inds))
    end
  end
  return igs
end

function neighboring_index_groups(contraction, index_groups)
  inds = noncommoninds(vectorize(contraction)...)
  nigs = []
  for ig in index_groups
    if issubset(ig.data, inds)
      push!(nigs, ig)
    end
  end
  return nigs
end

"""
get the index group information of the input tn
Return:
==========
ig_neighbor_dict: a dictionary that maps each indexgroup to its neighbors
tn_ig_dict: a dictionary that maps each tn to its list of index groups
"""
function index_group_info(tn::Vector)
  ig_neighbor_set = Set{Vector{IndexGroup}}()
  tn_ig_dict = Dict{Vector,Vector{IndexGroup}}()
  inds_ig_dict = Dict{Vector{Index},IndexGroup}()

  function update_neighbor(igs)
    if length(igs) > 1
      push!(ig_neighbor_set, igs)
    end
  end

  index_groups = get_index_groups(tn)
  contractions = find_topo_sort(tn)
  for c in contractions
    tn_ig_dict[c] = neighboring_index_groups(c, index_groups)
    if c isa Vector{<:Vector} && (isleaf(c[1]) || isleaf(c[2]))
      inter_igs = intersect(tn_ig_dict[c[1]], tn_ig_dict[c[2]])
      update_neighbor(inter_igs)
      if isleaf(c[1])
        update_neighbor(setdiff(tn_ig_dict[c[1]], inter_igs))
      end
      if isleaf(c[2])
        update_neighbor(setdiff(tn_ig_dict[c[2]], inter_igs))
      end
    end
  end
  return tn_ig_dict, ig_neighbor_set
end
