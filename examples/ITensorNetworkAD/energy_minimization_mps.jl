using ITensors
using ITensorNetworkAD
using OptimKit
using Random
using Zygote

using ITensors: data

Random.seed!(1243)

N = 20
χ = 20

s = siteinds("S=1/2", N; conserve_qns=true)
os = OpSum()
for n in 1:(N - 1)
  os .+= 0.5, "S+", n, "S-", n + 1
  os .+= 0.5, "S-", n, "S+", n + 1
  os .+= "Sz", n, "Sz", n + 1
end
Hmpo = MPO(os, s)
ψ₀mps = randomMPS(s, n -> isodd(n) ? "↑" : "↓"; linkdims=χ)

H = data(Hmpo)
ψ₀ = data(ψ₀mps)

# The Rayleigh quotient to minimize
function E(H::Vector{ITensor}, ψ::Vector{ITensor})
  N = length(ψ)
  ψdag = dag.(addtags.(ψ, "bra"; tags="Link"))
  ψ′dag = prime.(ψdag)
  e = ITensor(1.0)
  for n in 1:N
    e = e * ψ′dag[n] * H[n] * ψ[n]
  end
  norm = ITensor(1.0)
  for n in 1:N
    norm = norm * ψdag[n] * ψ[n]
  end
  return e[] / norm[]
end

@show sum(flux, ψ₀)
@show E(H, ψ₀)

E(ψ) = E(H, ψ)
∇E(ψ) = E'(ψ)
fg(ψ) = (E(ψ), ∇E(ψ))

linesearch = HagerZhangLineSearch(; c₁=0.1, c₂=0.9, ϵ=1e-6, θ=1 / 2, γ=2 / 3, ρ=5.0)
algorithm = LBFGS(5; maxiter=100, gradtol=1e-8, linesearch=linesearch, verbosity=2)
ψ, fψ, gψ, numfg, normgradhistory = optimize(fg, ψ₀, algorithm)

sweeps = Sweeps(5)
setmaxdim!(sweeps, χ)
fψ_dmrg, ψ_dmrg = dmrg(Hmpo, ψ₀mps, sweeps)

time_Eψ = @elapsed E(ψ)
time_∇Eψ = @elapsed E'(ψ)
@show time_∇Eψ / time_Eψ;
@show fψ_dmrg
@show E(H, ψ)
