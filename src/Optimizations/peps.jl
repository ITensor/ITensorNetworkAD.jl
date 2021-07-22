using AutoHOOT, ChainRulesCore, Zygote
using ..ITensorAutoHOOT
using ..ITensorNetworks
using ITensors: setinds
using ..ITensorNetworks: PEPS, inner_network, flatten, insert_projectors, split_network
using ..ITensorAutoHOOT: batch_tensor_contraction

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
    dP = ITensors.removetags(linkinds, P, args...)
    return (NoTangent(), NoTangent(), dP, broadcast_notangent(args)...)
  end
  return ITensors.addtags(linkinds, P, args...), pullback
end

function ChainRulesCore.rrule(
  ::typeof(ITensors.removetags), ::typeof(linkinds), P::PEPS, args...
)
  function pullback(dtag_peps)
    dP = ITensors.addtags(linkinds, P, args...)
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

"""Generate an array of networks representing inner products, <p|H_1|p>, ..., <p|H_n|p>, <p|p>
Parameters
----------
peps: a peps network with datatype PEPS
peps_prime: prime of peps used for inner products
peps_prime_ham: prime of peps used for calculating expectation values
Hlocal: An array of MPO operators with datatype LocalMPO
Returns
-------
An array of networks.
"""
function generate_inner_network(
  peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, Hlocal::Array
)
  network_list = []
  for H_term in Hlocal
    inner = inner_network(
      peps, peps_prime, peps_prime_ham, H_term.mpo, [H_term.coord1, H_term.coord2]
    )
    network_list = vcat(network_list, [inner])
  end
  inner = inner_network(peps, peps_prime)
  network_list = vcat(network_list, [inner])
  return network_list
end

# gradient of this function returns nothing.
@non_differentiable generate_inner_network(
  peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, Hlocal::Array
)

function generate_inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Array{<:ITensor,1},
  Hlocal::Array,
)
  network_list = generate_inner_network(peps, peps_prime, peps_prime_ham, Hlocal)
  return map(network -> vcat(network, projectors), network_list)
end

@non_differentiable generate_inner_network(
  peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, projectors::Array, Hlocal::Array
)

function rayleigh_quotient(inners::Array)
  self_inner = inners[length(inners)][]
  expectations = sum(inners[1:(length(inners) - 1)])[]
  return expectations / self_inner
end

function loss_grad_wrap(peps::PEPS, Hlocal::Array)
  function loss(peps::PEPS)
    peps_prime = prime(linkinds, peps)
    peps_prime_ham = prime(peps)
    network_list = generate_inner_network(peps, peps_prime, peps_prime_ham, Hlocal)
    variables = flatten([peps, peps_prime, peps_prime_ham])
    inners = batch_tensor_contraction(network_list, variables...)
    return rayleigh_quotient(inners)
  end
  loss_w_grad(peps::PEPS) = loss(peps), gradient(loss, peps)[1]
  return loss_w_grad
end

@non_differentiable insert_projectors(peps::PEPS, center)

@non_differentiable ITensors.commoninds(p1::PEPS, p2::PEPS)

function loss_grad_wrap(peps::PEPS, Hlocal::Array, ::typeof(insert_projectors))
  function loss(peps::PEPS)
    # insert projectors here
    center = (div(size(peps.data)[1], 2), :)
    _, projectors = insert_projectors(peps, center)
    peps_bra = addtags(linkinds, peps, "bra")
    peps_ket = addtags(linkinds, peps, "ket")
    sites = commoninds(peps_bra, peps_ket)
    peps_bra_split = split_network(peps_bra) #TODO: implement rrule
    peps_ket_split = split_network(peps_ket)
    peps_ket_split_ham = prime(sites, peps_ket_split) #TODO: implement rrule
    # generate network
    network_list = generate_inner_network(
      peps_bra_split, peps_ket_split, peps_ket_split_ham, projectors, Hlocal
    )
    variables = flatten([peps_bra_split, peps_ket_split, peps_ket_split_ham])
    inners = batch_tensor_contraction(network_list, variables...)
    return rayleigh_quotient(inners)
  end
  loss_w_grad(peps::PEPS) = loss(peps), gradient(loss, peps)[1]
  return loss_w_grad
end
