Base.isless(a::ITensors.Index, b::ITensors.Index) = a.id < b.id

function general_mps_MPSTensor(tensor::ITensor; cutoff, maxdim)
  mps_out = if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, sort(inds(tensor)); cutoff=cutoff, maxdim=maxdim)
  end
  return MPSTensor(mps_out)
end

function general_mps_contract(mps1::MPS, mps2::MPS; cutoff, maxdim)
  ## TODO
  tensor = contract(vcat(collect(mps1), collect(mps2))...)
  return if size(tensor) == ()
    MPS([tensor])
  else
    MPS(tensor, sort(inds(tensor)); cutoff=cutoff, maxdim=maxdim)
  end
end

function general_mps_contract(t1::MPSTensor, t2::MPSTensor; kwargs...)
  mps_out = general_mps_contract(t1.mps, t2.mps; kwargs...)
  return MPSTensor(mps_out)
end
