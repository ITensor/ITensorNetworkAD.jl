using Zygote, OptimKit
using ..ITensorAutoHOOT
using ..ITensorNetworks
using ..ITensorAutoHOOT: batch_tensor_contraction
using ..ITensorNetworks:
  PEPS,
  generate_inner_network,
  flatten,
  insert_projectors,
  split_network,
  rayleigh_quotient,
  Models
using ..ITensorNetworks: broadcast_add, broadcast_minus, broadcast_mul, broadcast_inner

function loss_grad_wrap(peps::PEPS, Hs::Array)
  function loss(peps::PEPS)
    peps_prime = prime(linkinds, peps)
    peps_prime_ham = prime(peps)
    network_list = generate_inner_network(peps, peps_prime, peps_prime_ham, Hs)
    variables = flatten([peps, peps_prime, peps_prime_ham])
    inners = batch_tensor_contraction(network_list, variables...)
    return rayleigh_quotient(inners)
  end
  loss_w_grad(peps::PEPS) = loss(peps), gradient(loss, peps)[1]
  return loss_w_grad
end

function loss_grad_wrap(
  peps::PEPS, Hs::Array, ::typeof(insert_projectors); cutoff=1e-15, maxdim=100
)
  center = (div(size(peps.data)[1] - 1, 2) + 1, :)
  init_call = true
  cache = NetworkCache()
  function loss(peps::PEPS)
    tn_split, projectors = insert_projectors(peps, center, cutoff, maxdim)
    peps_bra = addtags(linkinds, peps, "bra")
    peps_ket = addtags(linkinds, peps, "ket")
    sites = commoninds(peps_bra, peps_ket)
    peps_bra_split = split_network(peps_bra)
    peps_ket_split = split_network(peps_ket)
    peps_ket_split_ham = prime(sites, peps_ket_split)
    # generate network
    network_list = generate_inner_network(
      peps_bra_split, peps_ket_split, peps_ket_split_ham, projectors, Hs
    )
    variables = flatten([peps_bra_split, peps_ket_split, peps_ket_split_ham])
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
    network_list_row = generate_inner_network(
      peps_bra_split, peps_ket_split, peps_ket_split_ham, projectors_row, Hs_row
    )
    network_list_column = generate_inner_network(
      peps_bra_split_rot,
      peps_ket_split_rot,
      peps_ket_split_rot_ham,
      projectors_column,
      Hs_column,
    )
    network_inner = generate_inner_network(
      peps_bra_split, peps_ket_split, peps_ket_split_ham, projectors_row[1], []
    )
    network_list = vcat(network_list_row, network_list_column, network_inner)
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

function gd_error_tracker(
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
  peps::PEPS, Hs::Array; stepsize::Float64, num_sweeps::Int, cutoff=1e-15, maxdim=100
)
  loss_w_grad = loss_grad_wrap(peps, Hs)
  loss_w_grad_approx = loss_grad_wrap(
    peps, Hs, insert_projectors; cutoff=cutoff, maxdim=maxdim
  )
  return gd_error_tracker(
    loss_w_grad,
    loss_w_grad_approx;
    stepsize=stepsize,
    num_sweeps=num_sweeps,
    cutoff=cutoff,
    maxdim=maxdim,
  )
end

function gd_error_tracker(
  peps::PEPS,
  Hs::Array{Models.LineMPO},
  stepsize::Float64,
  num_sweeps::Int,
  cutoff=1e-15,
  maxdim=100,
)
  Hs_row = [H for H in Hs if H.coord[2] isa Colon]
  Hs_column = [H for H in Hs if H.coord[1] isa Colon]
  loss_w_grad = loss_grad_wrap(peps, vcat(Hs_row, Hs_column))
  loss_w_grad_approx = loss_grad_wrap(
    peps, Hs_row, Hs_column, insert_projectors; cutoff=cutoff, maxdim=maxdim
  )
  return gd_error_tracker(
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
  linesearch = HagerZhangLineSearch()
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
