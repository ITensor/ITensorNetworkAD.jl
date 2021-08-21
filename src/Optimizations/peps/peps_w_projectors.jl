using Zygote, OptimKit
using ..ITensorAutoHOOT
using ..ITensorNetworks
using ..ITensorAutoHOOT: batch_tensor_contraction
using ..ITensorNetworks:
  PEPS,
  inner_network,
  inner_networks,
  flatten,
  insert_projectors,
  split_network,
  rayleigh_quotient,
  Models,
  tree

function loss_grad_wrap(
  peps::PEPS,
  Hs::Array{Models.LineMPO},
  ::typeof(insert_projectors);
  cutoff=1e-15,
  maxdim=100,
)
  Hs_row = [H for H in Hs if H.coord[2] isa Colon]
  Hs_column = [H for H in Hs if H.coord[1] isa Colon]
  init_call = true
  cache = NetworkCache()
  function loss(peps::PEPS)
    tn_split_row, tn_split_column, projectors_row, projectors_column = insert_projectors(
      peps, cutoff, maxdim
    )
    peps_bra = addtags(linkinds, peps, "bra")
    peps_ket = addtags(linkinds, peps, "ket")
    peps_bra_rot = addtags(linkinds, peps, "brarot")
    peps_ket_rot = addtags(linkinds, peps, "ketrot")
    sites = commoninds(peps_bra, peps_ket)
    peps_bra_split = split_network(peps_bra)
    peps_ket_split = split_network(peps_ket)
    peps_ket_split_ham = prime(sites, peps_ket_split)
    peps_bra_split_rot = split_network(peps_bra_rot, true)
    peps_ket_split_rot = split_network(peps_ket_rot, true)
    peps_ket_split_rot_ham = prime(sites, peps_ket_split_rot)
    # generate network
    network_list_row = inner_networks(
      peps_bra_split, peps_ket_split, peps_ket_split_ham, projectors_row, Hs_row, tree
    )
    network_list_column = inner_networks(
      peps_bra_split_rot,
      peps_ket_split_rot,
      peps_ket_split_rot_ham,
      projectors_column,
      Hs_column,
      tree,
    )
    network_inner = inner_network(peps_bra_split, peps_ket_split, projectors_row[1], tree)
    network_list = vcat(network_list_row, network_list_column, [network_inner])
    variables = flatten([
      peps_bra_split,
      peps_ket_split,
      peps_ket_split_ham,
      peps_bra_split_rot,
      peps_ket_split_rot,
      peps_ket_split_rot_ham,
    ])
    if init_call == true
      cache = NetworkCache(network_list)
      init_call = false
    end
    inners = batch_tensor_contraction(network_list, cache, variables...)
    return rayleigh_quotient(inners)
  end
  loss_w_grad(peps::PEPS) = loss(peps), gradient(loss, peps)[1]
  return loss_w_grad
end

function gradient_descent(
  peps::PEPS,
  Hs::Array,
  ::typeof(insert_projectors);
  stepsize::Float64,
  num_sweeps::Int,
  cutoff=1e-15,
  maxdim=100,
)
  loss_w_grad = loss_grad_wrap(peps, Hs, insert_projectors; cutoff=cutoff, maxdim=maxdim)
  return gradient_descent(peps, loss_w_grad; stepsize=stepsize, num_sweeps=num_sweeps)
end

function gradient_descent(
  peps::PEPS,
  Hs::Array,
  ::typeof(insert_projectors),
  ::typeof(backtracking_linesearch);
  beta::Float64,
  num_sweeps::Int,
  cutoff=1e-15,
  maxdim=100,
)
  loss_w_grad = loss_grad_wrap(peps, Hs, insert_projectors; cutoff=cutoff, maxdim=maxdim)
  return gradient_descent(
    peps, loss_w_grad, backtracking_linesearch; beta=beta, num_sweeps=num_sweeps
  )
end

function OptimKit.optimize(
  peps::PEPS,
  Hs::Array,
  ::typeof(insert_projectors);
  num_sweeps::Int,
  method="GD",
  cutoff=1e-15,
  maxdim=100,
)
  loss_w_grad = loss_grad_wrap(peps, Hs, insert_projectors; cutoff=cutoff, maxdim=maxdim)
  return optimize(peps, loss_w_grad; num_sweeps=num_sweeps, method=method)
end

function gd_error_tracker(
  peps::PEPS, Hs::Vector; stepsize::Float64, num_sweeps::Int, cutoff=1e-15, maxdim=100
)
  ITensors.set_warn_order(40)
  loss_w_grad = loss_grad_wrap(peps, Hs)
  loss_w_grad_approx = loss_grad_wrap(
    peps, Hs, insert_projectors; cutoff=cutoff, maxdim=maxdim
  )
  return gd_error_tracker(
    peps,
    loss_w_grad,
    loss_w_grad_approx;
    stepsize=stepsize,
    num_sweeps=num_sweeps,
    cutoff=cutoff,
    maxdim=maxdim,
  )
end
