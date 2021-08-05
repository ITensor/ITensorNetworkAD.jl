using AutoHOOT, ChainRulesCore

broadcast_notangent(a) = broadcast(_ -> NoTangent(), a)

function ChainRulesCore.rrule(
  ::typeof(split_network),
  tn::Matrix{ITensor};
  projector_center=default_projector_center(tn),
  rotation=false,
)
  function pullback(dtn_split::Matrix{ITensor})
    dtn = map(t -> replaceprime(t, 1 => 0), dtn_split)
    return (NoTangent(), dtn, NoTangent(), NoTangent())
  end
  return split_network(tn; projector_center=projector_center, rotation=rotation), pullback
end

function ChainRulesCore.rrule(::typeof(ITensors.data), P::PEPS)
  return P.data, d_data -> (NoTangent(), PEPS(d_data))
end

function ChainRulesCore.rrule(::typeof(PEPS), data::Matrix{ITensor})
  return PEPS(data), dpeps -> (NoTangent(), dpeps.data)
end

function ChainRulesCore.rrule(::typeof(ITensors.prime), P::PEPS, n::Integer=1)
  return prime(P, n), dprime -> (NoTangent(), prime(dprime, -n), NoTangent())
end

function ChainRulesCore.rrule(
  ::typeof(ITensors.addtags), ::typeof(linkinds), P::PEPS, args...
)
  function pullback(dtag_peps)
    dP = ITensors.removetags(linkinds, dtag_peps, args...)
    return (NoTangent(), NoTangent(), dP, broadcast_notangent(args)...)
  end
  return ITensors.addtags(linkinds, P, args...), pullback
end

function ChainRulesCore.rrule(
  ::typeof(ITensors.removetags), ::typeof(linkinds), P::PEPS, args...
)
  function pullback(dtag_peps)
    dP = ITensors.addtags(linkinds, dtag_peps, args...)
    return (NoTangent(), NoTangent(), dP, broadcast_notangent(args)...)
  end
  return ITensors.removetags(linkinds, P, args...), pullback
end

function ChainRulesCore.rrule(
  ::typeof(ITensors.prime), ::typeof(linkinds), P::PEPS, n::Integer=1
)
  return prime(linkinds, P, n),
  dprime -> (NoTangent(), NoTangent(), prime(linkinds, dprime, -n), NoTangent())
end

function ChainRulesCore.rrule(
  ::typeof(ITensors.prime), indices::Array{<:Index,1}, P::PEPS, n::Integer=1
)
  primeinds = [prime(ind, n) for ind in indices]
  return prime(indices, P, n),
  dprime -> (NoTangent(), NoTangent(), prime(primeinds, dprime, -n), NoTangent())
end

function ChainRulesCore.rrule(::typeof(flatten), v::Array{<:PEPS})
  size_list = [size(peps.data) for peps in v]
  function adjoint_pullback(dt)
    dt = [t for t in dt]
    index = 0
    dv = []
    for (dimy, dimx) in size_list
      size = dimy * dimx
      d_peps = PEPS(reshape(dt[(index + 1):(index + size)], dimy, dimx))
      index += size
      push!(dv, d_peps)
    end
    return (NoTangent(), dv)
  end
  return flatten(v), adjoint_pullback
end

# gradient of this function returns nothing.
@non_differentiable generate_inner_network(
  peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, Hs::Array
)

@non_differentiable generate_inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{<:ITensor},
  Hs::Array,
)

@non_differentiable generate_inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{Vector{ITensor}},
  Hs::Vector{Tuple},
)

@non_differentiable insert_projectors(peps::PEPS, cutoff, maxdim)

@non_differentiable insert_projectors(peps::PEPS, center::Tuple, cutoff, maxdim)

@non_differentiable ITensors.commoninds(p1::PEPS, p2::PEPS)

@non_differentiable SubNetwork(inputs::Union{SubNetwork,ITensor}...)
