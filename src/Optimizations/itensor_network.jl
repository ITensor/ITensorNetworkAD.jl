using ChainRulesCore
using ..ITensorNetworks
using ..ITensorNetworks: split_network, default_projector_center

function ChainRulesCore.rrule(
  ::typeof(split_network),
  tn::Matrix{ITensor};
  projector_center=default_projector_center(tn),
)
  dimy, dimx = size(tn)
  function pullback(dtn_split::Matrix{ITensor})
    dtn = copy(dtn_split)
    for ii in 1:dimy
      for jj in 1:dimx
        dt, t = dtn[ii, jj], tn[ii, jj]
        indices = inds(t)
        get_index(i_split) = findfirst(x -> x.id == i_split.id, indices)
        indices_reorder = [indices[get_index(i_split)] for i_split in inds(dt)]
        dtn[ii, jj] = setinds(dt, Tuple(indices_reorder))
      end
    end
    return (NoTangent(), dtn, NoTangent())
  end
  return split_network(tn; projector_center=projector_center), pullback
end
