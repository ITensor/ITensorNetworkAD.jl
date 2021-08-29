function singleindex_mps_MPSTensor(tensor::ITensor; cutoff, maxdim)
  mps_out = if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, inds(tensor); cutoff=cutoff, maxdim=maxdim)
  end
  return MPSTensor(mps_out)
end

function singleindex_mps_contract(mps1::MPS, mps2::MPS; cutoff, maxdim)
  ## TODO: modify this function based on https://arxiv.org/pdf/1912.03014.pdf
  tensor = contract(vcat(collect(mps1), collect(mps2))...)
  return if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, inds(tensor); cutoff=cutoff, maxdim=maxdim)
  end
end

function singleindex_mps_contract(t1::MPSTensor, t2::MPSTensor; kwargs...)
  mps_out = singleindex_mps_contract(t1.mps, t2.mps; kwargs...)
  return MPSTensor(mps_out)
end
