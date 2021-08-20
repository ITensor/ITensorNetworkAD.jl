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
using ..ITensorNetworks: broadcast_add, broadcast_minus, broadcast_mul, broadcast_inner

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

function backtracking_linesearch(beta::Float64, loss_w_grad, peps)
  stepsize = 1.0
  stepsize_lb = 0.0
  l, g = loss_w_grad(peps)
  update_peps = broadcast_minus(peps, broadcast_mul(stepsize, g))
  update_l, update_g = loss_w_grad(update_peps)
  gnrm_square = broadcast_inner(g, g)
  threshold = l - stepsize / 2.0 * broadcast_inner(g, g)
  print("threshold: $threshold, gradient norm: $gnrm_square \n")
  while update_l > threshold && stepsize >= stepsize_lb
    stepsize = stepsize * beta
    print("stepsize trial: $stepsize \n")
    update_peps = broadcast_minus(peps, broadcast_mul(stepsize, g))
    update_l, update_g = loss_w_grad(update_peps)
    threshold = l - stepsize / 2.0 * broadcast_inner(g, g)
    diffg = broadcast_minus(g, update_g)
    update_gnrm_square = broadcast_inner(diffg, diffg)
    print("threshold: $threshold, update_l: $update_l, update_g: $update_gnrm_square \n")
  end
  print("stepsize from backtracking_linesearch is $stepsize \n")
  return stepsize
end

function gradient_descent(peps::PEPS, loss_w_grad; stepsize::Float64, num_sweeps::Int)
  # gradient descent iterations
  losses = []
  for iter in 1:num_sweeps
    l, g = loss_w_grad(peps)
    print("The rayleigh quotient at iteraton $iter is $l\n")
    peps = broadcast_minus(peps, broadcast_mul(stepsize, g))
    push!(losses, l)
  end
  return losses
end

function gradient_descent(
  peps::PEPS, loss_w_grad, ::typeof(backtracking_linesearch); beta::Float64, num_sweeps::Int
)
  # gradient descent iterations
  losses = []
  for iter in 1:num_sweeps
    l, g = loss_w_grad(peps)
    print("The rayleigh quotient at iteraton $iter is $l\n")
    stepsize = backtracking_linesearch(beta, loss_w_grad, peps)
    peps = broadcast_minus(peps, broadcast_mul(stepsize, g))
    push!(losses, l)
  end
  return losses
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

function gd_error_tracker(
  peps,
  loss_w_grad,
  loss_w_grad_approx;
  stepsize::Float64,
  num_sweeps::Int,
  cutoff=1e-15,
  maxdim=100,
)
  for iter in 1:num_sweeps
    l, g = loss_w_grad(peps)
    l_approx, g_approx = loss_w_grad_approx(peps)
    g_diff = broadcast_minus(g, g_approx)
    g_diff_nrm = broadcast_inner(g_diff, g_diff)
    print("The gradient difference norm at iteraton $iter is $g_diff_nrm\n")
    print("The rayleigh quotient at iteraton $iter is $l\n")
    print("The approximate rayleigh quotient at iteraton $iter is $l_approx\n")
    peps = broadcast_minus(peps, broadcast_mul(stepsize, g))
  end
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

function OptimKit.optimize(peps::PEPS, loss_w_grad; num_sweeps::Int, method="GD")
  @assert(method in ["GD", "LBFGS", "CG"])
  inner(x, peps1, peps2) = broadcast_inner(peps1, peps2)
  scale(peps, alpha) = broadcast_mul(alpha, peps)
  add(peps1, peps2, alpha) = broadcast_add(peps1, broadcast_mul(alpha, peps2))
  retract(peps1, peps2, alpha) = (add(peps1, peps2, alpha), peps2)
  linesearch = HagerZhangLineSearch(; c₁=0.1, c₂=0.9, verbosity=0)
  if method == "GD"
    alg = GradientDescent(num_sweeps, 1e-8, linesearch, 2)
  elseif method == "LBFGS"
    alg = LBFGS(16; maxiter=num_sweeps, gradtol=1e-8, linesearch=linesearch, verbosity=2)
  elseif method == "CG"
    alg = ConjugateGradient(;
      maxiter=num_sweeps, gradtol=1e-8, linesearch=linesearch, verbosity=2
    )
  end
  _, _, _, _, history = OptimKit.optimize(
    loss_w_grad, peps, alg; inner=inner, (scale!)=scale, (add!)=add, retract=retract
  )
  return history[:, 1]
end

function OptimKit.optimize(peps::PEPS, Hs::Array; num_sweeps::Int, method="GD")
  loss_w_grad = loss_grad_wrap(peps, Hs)
  return optimize(peps, loss_w_grad; num_sweeps=num_sweeps, method=method)
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
