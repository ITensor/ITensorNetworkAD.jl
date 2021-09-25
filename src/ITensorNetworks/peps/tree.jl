using ..ITensorAutoHOOT
using ..ITensorAutoHOOT: SubNetwork, neighboring_tensors

"""Returns a tree structure for a line of tensors with projectors
Parameters
----------
line_size: size of the line structure
center_index: the center index of the line
site_tensors: a function, site_tensors(i) returns a list of tensors at position i
projectors: the projectors of the line structure
Returns
-------
Two trees, one in front of the center_index and another one after center_index
Example
-------
   |  |   |
p1-p2 |  p3-p4
 | |  |   | |
 | |  |   | |
s1-s2-s3-s4-s5
here line_size=5, center_index=3, si represents the list of tensors returned by site_tensors(i),
projectors are [p1, p2, p3, p4].
Returns two trees: [[s1, p1], s2, p2] and [[s5, p4], s4, p3]
"""
function tree(line_size, center_index, site_tensors, projectors::Vector{ITensor})
  front_tree, back_tree = nothing, nothing
  for i in 1:(center_index - 1)
    connect_projectors = neighboring_tensors(SubNetwork(site_tensors(i)), projectors)
    if front_tree == nothing
      inputs = vcat(site_tensors(i), connect_projectors)
    else
      inputs = vcat(site_tensors(i), connect_projectors, [front_tree])
    end
    front_tree = SubNetwork(inputs)
  end
  for i in line_size:-1:(center_index + 1)
    connect_projectors = neighboring_tensors(SubNetwork(site_tensors(i)), projectors)
    if back_tree == nothing
      inputs = vcat(site_tensors(i), connect_projectors)
    else
      inputs = vcat(site_tensors(i), connect_projectors, [back_tree])
    end
    back_tree = SubNetwork(inputs)
  end
  return front_tree, back_tree
end

function tree(sub_peps_bra::Vector, sub_peps_ket::Vector, projectors::Vector{ITensor})
  out_inds = inds(SubNetwork(vcat(sub_peps_bra, sub_peps_ket, projectors)))
  is_neighbor(t) = length(intersect(out_inds, inds(t))) > 0
  center_index = [i for (i, t) in enumerate(sub_peps_bra) if is_neighbor(t)]
  @assert length(center_index) == 1
  center_index = center_index[1]
  @assert is_neighbor(sub_peps_ket[center_index])
  site_tensors(i) = [sub_peps_bra[i], sub_peps_ket[i]]
  front_tree, back_tree = tree(
    length(sub_peps_bra), center_index, site_tensors, projectors::Vector{ITensor}
  )
  return SubNetwork([
    sub_peps_bra[center_index], sub_peps_ket[center_index], front_tree, back_tree
  ])
end

function tree(
  sub_peps_bra::Vector, sub_peps_ket::Vector, mpo::Vector, projectors::Vector{ITensor}
)
  out_inds = inds(SubNetwork(vcat(sub_peps_bra, sub_peps_ket, mpo, projectors)))
  is_neighbor(t) = length(intersect(out_inds, inds(t))) > 0
  center_index = [i for (i, t) in enumerate(sub_peps_bra) if is_neighbor(t)]
  @assert length(center_index) == 1
  center_index = center_index[1]
  @assert is_neighbor(sub_peps_ket[center_index])
  site_tensors(i) = [sub_peps_bra[i], sub_peps_ket[i], mpo[i]]
  front_tree, back_tree = tree(
    length(sub_peps_bra), center_index, site_tensors, projectors::Vector{ITensor}
  )
  return SubNetwork([
    sub_peps_bra[center_index],
    sub_peps_ket[center_index],
    mpo[center_index],
    front_tree,
    back_tree,
  ])
end
