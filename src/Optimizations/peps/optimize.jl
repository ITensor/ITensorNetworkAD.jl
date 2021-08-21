using OptimKit
using ..ITensorNetworks
using ..ITensorNetworks:
  PEPS, broadcast_add, broadcast_minus, broadcast_mul, broadcast_inner

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
