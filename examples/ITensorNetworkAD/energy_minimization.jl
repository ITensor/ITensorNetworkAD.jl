using ITensors
using ITensorNetworkAD
using OptimKit
using Random
using Zygote

Random.seed!(1243)

N = 6
s = siteinds("S=1/2", N; conserve_qns=true)
os = OpSum()
for n in 1:(N - 1)
  os .+= 0.5, "S+", n, "S-", n+1
  os .+= 0.5, "S-", n, "S+", n+1
  os .+= "Sz", n, "Sz", n+1
end
Hmpo = MPO(os, s)
ψ₀mps = randomMPS(s, n -> isodd(n) ? "↑" : "↓")
H = prod(Hmpo)
ψ₀ = prod(ψ₀mps)

# The Rayleigh quotient to minimize
function E(H::ITensor, ψ::ITensor)
  ψdag = dag(ψ)
  return (ψdag' * H * ψ)[] / (ψdag * ψ)[]
end

@show flux(ψ₀)
@show E(H, ψ₀)

E(ψ::ITensor) = E(H, ψ)
∇E(ψ::ITensor) = E'(ψ)
fg(ψ::ITensor) = (E(ψ), ∇E(ψ))

linesearch = HagerZhangLineSearch(; c₁=.1, c₂=.9, ϵ=1e-6, θ=1/2, γ=2/3, ρ=5., verbosity=0)
algorithm = LBFGS(3; maxiter=30, gradtol=1e-8, linesearch=linesearch)
ψ, fψ, gψ, numfg, normgradhistory = optimize(fg, ψ₀, algorithm)

display(normgradhistory)

@show E(H, ψ)

D, _ = eigen(H; ishermitian=true)
@show minimum(D)

