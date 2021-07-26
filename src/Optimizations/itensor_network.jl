using ChainRulesCore
using ..ITensorNetworks
using ..ITensorNetworks: split_network, default_projector_center

inv_op(::typeof(addtags)) = removetags
inv_op(::typeof(removetags)) = addtags

function ChainRulesCore.rrule(
  ::typeof(split_network),
  tn::Matrix{ITensor};
  projector_center=default_projector_center(tn),
)
  dimy, dimx = size(tn)
  tn_vec = vec(tn)
  function pullback(dtn_split::Matrix{ITensor})
    dtn_split_vec = vec(dtn_split)
    dtn_vec = []
    for i in 1:(dimy * dimx)
      indices = inds(tn_vec[i])
      indices_reorder = []
      for i_split in inds(dtn_split_vec[i])
        index = findall(x -> x.id == i_split.id, indices)
        @assert(length(index) == 1)
        push!(indices_reorder, indices[index[1]])
      end
      push!(dtn_vec, setinds(dtn_split_vec[i], Tuple(indices_reorder)))
    end
    dtn = reshape(dtn_vec, (dimy, dimx))
    return (NoTangent(), dtn, NoTangent())
  end
  return split_network(tn; projector_center=projector_center), pullback
end
