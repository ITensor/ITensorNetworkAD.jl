mutable struct IndexGroup
  data::Vector
  istree::Bool
end

# TODO: general tags are not comparable
Base.isless(a::Index, b::Index) = id(a) < id(b) || (id(a) == id(b) && plev(a) < plev(b)) # && tags(a) < tags(b)

function IndexGroup(indices::Vector{<:Index})
  return IndexGroup(sort(indices), false)
end

@profile function get_index_groups(tn_tree::Vector)
  tn_leaves = get_leaves(tn_tree)
  igs = []
  for (t1, t2) in powerset(tn_leaves, 2, 2)
    inds = intersect(noncommoninds(t1...), noncommoninds(t2...))
    if length(inds) >= 1
      push!(igs, IndexGroup(inds))
    end
  end
  return igs
end

@profile function neighbor_index_groups(contraction, index_groups)
  inds = noncommoninds(vectorize(contraction)...)
  nigs = []
  for ig in index_groups
    if issubset(ig.data, inds)
      push!(nigs, ig)
    end
  end
  return nigs
end
