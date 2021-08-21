using Zygote, OptimKit
using ..ITensorAutoHOOT
using ..ITensorNetworks
using ..ITensorAutoHOOT: batch_tensor_contraction
using ..ITensorNetworks: PEPS, inner_network, inner_networks, flatten, rayleigh_quotient

function loss_grad_wrap(peps::PEPS, Hs::Array)
  function loss(peps::PEPS)
    peps_bra = addtags(linkinds, peps, "bra")
    peps_ket = addtags(linkinds, peps, "ket")
    sites = commoninds(peps_bra, peps_ket)
    peps_ket_ham = prime(sites, peps_ket)
    network_H = inner_networks(peps_bra, peps_ket, peps_ket_ham, Hs)
    network_inner = inner_network(peps_bra, peps_ket)
    network_list = vcat(network_H, [network_inner])
    variables = flatten([peps_bra, peps_ket, peps_ket_ham])
    inners = batch_tensor_contraction(network_list, variables...)
    return rayleigh_quotient(inners)
  end
  loss_w_grad(peps::PEPS) = loss(peps), gradient(loss, peps)[1]
  return loss_w_grad
end

"""Update PEPS based on gradient descent
Parameters
----------
peps: a peps network with datatype PEPS
Hs: An array of MPO operators with datatype LocalMPO or LineMPO
stepsize: step size used in the gradient descent
num_sweeps: number of gradient descent sweeps/iterations
Returns
-------
An array containing Rayleigh quotient losses after each iteration.
"""
function gradient_descent(peps::PEPS, Hs::Array; stepsize::Float64, num_sweeps::Int)
  loss_w_grad = loss_grad_wrap(peps, Hs)
  return gradient_descent(peps, loss_w_grad; stepsize=stepsize, num_sweeps=num_sweeps)
end

function OptimKit.optimize(peps::PEPS, Hs::Array; num_sweeps::Int, method="GD")
  loss_w_grad = loss_grad_wrap(peps, Hs)
  return optimize(peps, loss_w_grad; num_sweeps=num_sweeps, method=method)
end
