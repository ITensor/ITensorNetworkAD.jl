using Zygote, OptimKit
using ..ITensorAutoHOOT
using ..ITensorNetworks
using ..ITensorAutoHOOT: batch_tensor_contraction
using ..ITensorNetworks: PEPS, inner_network, inner_networks, flatten, rayleigh_quotient

function loss_grad_wrap(peps::PEPS, Hs::Array, tensortype; kwargs...)
  function loss(peps::PEPS)
    peps_bra = addtags(linkinds, peps, "bra")
    peps_ket = addtags(linkinds, peps, "ket")
    sites = commoninds(peps_bra, peps_ket)
    peps_ket_ham = prime(sites, peps_ket)
    network_H = inner_networks(peps_bra, peps_ket, peps_ket_ham, Hs)
    network_inner = inner_network(peps_bra, peps_ket)
    network_list = vcat(network_H, [network_inner])
    variables = flatten([peps_bra, peps_ket, peps_ket_ham])
    inners = batch_tensor_contraction(tensortype, network_list, variables...; kwargs...)
    return rayleigh_quotient(inners)
  end
  loss_w_grad(peps::PEPS) = loss(peps), gradient(loss, peps)[1]
  return loss_w_grad
end

function gradient_descent(
  peps::PEPS, Hs::Array, tensortype; stepsize::Float64, num_sweeps::Int, kwargs...
)
  loss_w_grad = loss_grad_wrap(peps, Hs, tensortype; kwargs...)
  return gradient_descent(peps, loss_w_grad; stepsize=stepsize, num_sweeps=num_sweeps)
end
