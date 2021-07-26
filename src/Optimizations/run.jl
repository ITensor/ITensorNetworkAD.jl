using OptimKit
using ..ITensorNetworks
using ..ITensorNetworks: broadcast_add, broadcast_minus, broadcast_mul, broadcast_inner

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
Hlocal: An array of MPO operators with datatype LocalMPO
stepsize: step size used in the gradient descent
num_sweeps: number of gradient descent sweeps/iterations
Returns
-------
An array containing Rayleigh quotient losses after each iteration.
"""
function gradient_descent(peps::PEPS, Hlocal::Array; stepsize::Float64, num_sweeps::Int)
  loss_w_grad = loss_grad_wrap(peps, Hlocal)
  return gradient_descent(peps, loss_w_grad; stepsize=stepsize, num_sweeps=num_sweeps)
end

function gradient_descent(
  peps::PEPS, Hlocal::Array, ::typeof(insert_projectors); stepsize::Float64, num_sweeps::Int
)
  loss_w_grad = loss_grad_wrap(peps, Hlocal, insert_projectors)
  return gradient_descent(peps, loss_w_grad; stepsize=stepsize, num_sweeps=num_sweeps)
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

function OptimKit.optimize(peps::PEPS, Hlocal::Array; num_sweeps::Int, method="GD")
  loss_w_grad = loss_grad_wrap(peps, Hlocal)
  return optimize(peps, loss_w_grad; num_sweeps=num_sweeps, method=method)
end

function OptimKit.optimize(
  peps::PEPS, Hlocal::Array, ::typeof(insert_projectors); num_sweeps::Int, method="GD"
)
  loss_w_grad = loss_grad_wrap(peps, Hlocal, insert_projectors)
  return optimize(peps, loss_w_grad; num_sweeps=num_sweeps, method=method)
end
