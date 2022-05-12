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
    @info inds
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

# function index_group_info(tn::Vector)
#   ig_neighbor_set = Set{Vector{IndexGroup}}()
#   tn_ig_dict = Dict{Vector, Vector{IndexGroup}}()
#   inds_ig_dict = Dict{Vector{Index}, IndexGroup}()

#   function build_index_group(inds)
#     sort_inds = sort(inds)
#     if haskey(inds_ig_dict, sort_inds)
#       return inds_ig_dict[sort_inds]
#     end
#     return inds_ig_dict[sort_inds] = IndexGroup(sort_inds)
#   end

#   function inds_group_children(tnet1, tnet2, inds_list)
#     @assert isleaf(tnet2)
#     left_inds, right_inds = uncontractinds(tnet1), uncontractinds(tnet2)
#     inter_inds = intersect(left_inds, right_inds)
#     # form inds_groups of tnet[1] and tnet[2]
#     inds_list_left = subtree(inds_list, left_inds)
#     inds_list_right = subtree(inds_list, right_inds)
#     inds_list_left = merge_tree(inds_list_left, inter_inds; append=true)
#     inds_list_right = merge_tree(inds_list_right, inter_inds; append=true)
#     @assert length(vectorize(inds_list_left)) == length(left_inds) &&
#       length(vectorize(inds_list_right)) == length(right_inds)
#     inds_group_left = recur(tnet1, inds_list_left)
#     inds_group_right = recur(tnet2, inds_list_right)
#     return inds_group_left, inds_group_right
#   end

#   function recur(tnet::Vector{ITensor}, inds_list)
#     if inds_list isa Vector{<:Index}
#       return tn_ig_dict[tnet] = [build_index_group(inds_list)]
#     end
#     return tn_ig_dict[tnet] = [build_index_group(inds) for inds in inds_list]
#   end

#   function recur(tnet::Vector, inds_list)
#     @assert length(tnet) == 2
#     @assert isleaf(tnet[1]) || isleaf(tnet[2])
#     if isleaf(tnet[1]) && !(isleaf(tnet[2]))
#       return recur([tnet[2], tnet[1]], inds_list)
#     end
#     inds_group_left, inds_group_right = inds_group_children(tnet[1], tnet[2], inds_list)
#     @info "leftgroup", inds_group_left
#     @info "rightgroup", inds_group_right
#     tn_ig_dict[tnet] = collect(Set([inds_group_left..., inds_group_right...]))
#     @info "sdf", tn_ig_dict[tnet]
#     inter_igs = intersect(inds_group_left, inds_group_right)
#     tn_ig_dict[tnet] = setdiff(tn_ig_dict[tnet], inter_igs)
#     # update ig_neighbor_set
#     update_neighbor(inter_igs)
#     update_neighbor(setdiff(inds_group_right, inter_igs))
#     if isleaf(tnet[1])
#       update_neighbor(setdiff(inds_group_left, inter_igs))
#     end
#     return tn_ig_dict[tnet]
#   end

#   recur(tn, noncommoninds(vectorize(tn)...))
#   return tn_ig_dict, ig_neighbor_set
# end
