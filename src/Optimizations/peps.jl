using AutoHOOT, ChainRulesCore, Zygote
using ..ITensorAutoHOOT
using ..ITensorNetworks
using ITensors: setinds
using ..ITensorNetworks: PEPS, inner_network, extract_data
using ..ITensorAutoHOOT: batch_tensor_contraction

# TODO: rewrite this function
function ChainRulesCore.rrule(::typeof(ITensors.prime), peps::PEPS; ham=true)
  dimy, dimx = size(peps.data)
  peps_vec = vec(peps.data)
  function adjoint_pullback(dpeps_prime::PEPS)
    dpeps_prime_vec = vec(dpeps_prime.data)
    dpeps_vec = []
    for i in 1:(dimy * dimx)
      indices = inds(peps_vec[i])
      indices_reorder = []
      for i_prime in inds(dpeps_prime_vec[i])
        index = findall(x -> x.id == i_prime.id, indices)
        @assert(length(index) == 1)
        push!(indices_reorder, indices[index[1]])
      end
      push!(dpeps_vec, setinds(dpeps_prime_vec[i], Tuple(indices_reorder)))
    end
    dpeps = PEPS(reshape(dpeps_vec, (dimy, dimx)))
    return (NoTangent(), dpeps, NoTangent())
  end
  return prime(peps; ham=ham), adjoint_pullback
end

function ChainRulesCore.rrule(::typeof(extract_data), v::Array{<:PEPS})
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
  return extract_data(v), adjoint_pullback
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

function rayleigh_quotient(inners::Array)
  self_inner = inners[length(inners)][]
  expectations = sum(inners[1:(length(inners) - 1)])[]
  return expectations / self_inner
end

function loss_grad_wrap(peps::PEPS, Hlocal::Array)
  function loss(peps::PEPS)
    peps_prime = prime(peps; ham=false)
    peps_prime_ham = prime(peps; ham=true)
    network_list = generate_inner_network(peps, peps_prime, peps_prime_ham, Hlocal)
    variables = extract_data([peps, peps_prime, peps_prime_ham])
    inners = batch_tensor_contraction(network_list, variables...)
    return rayleigh_quotient(inners)
  end
  loss_w_grad(peps::PEPS) = loss(peps), gradient(loss, peps)[1]
  return loss_w_grad
end
