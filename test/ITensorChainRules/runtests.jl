using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using ITensors
using ITensors.NDTensors
using ITensorNetworkAD
using OptimKit
using Random
using Test
using Zygote

using Zygote: ZygoteRuleConfig

#
# ITensor extensions
#

#
# For ITensor compatibility with FiniteDifferences
#

function FiniteDifferences.to_vec(A::ITensor)
  # TODO: generalize to sparse tensors
  # TODO: define `itensor([1.0])` as well
  # as `itensor([1.0], ())` to help with generic code.
  function vec_to_ITensor(x)
    return isempty(inds(A)) ? ITensor(x[]) : itensor(x, inds(A))
  end
  return vec(array(A)), vec_to_ITensor
end

function FiniteDifferences.to_vec(x::Index)
  return (Bool[], _ -> x)
end

function FiniteDifferences.to_vec(x::Tuple{Vararg{Index}})
  return (Bool[], _ -> x)
end

function FiniteDifferences.to_vec(x::Pair{<:Tuple{Vararg{Index}},<:Tuple{Vararg{Index}}})
  return (Bool[], _ -> x)
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::ITensor)
  # TODO: generalize to sparse tensors
  return isempty(inds(A)) ? ITensor(randn(eltype(A))) : randomITensor(eltype(A), inds(A))
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::Tensor)
  # TODO: generalize to sparse tensors
  return randomTensor(eltype(A), inds(A))
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, x::Index)
  return NoTangent()
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, x::Tuple{Vararg{Index}})
  return NoTangent()
end

function FiniteDifferences.rand_tangent(
  rng::AbstractRNG, x::Pair{<:Tuple{Vararg{Index}},<:Tuple{Vararg{Index}}}
)
  return NoTangent()
end

#
# For ITensor compatibility with ChainRulesTestUtils
#

function ChainRulesTestUtils.test_approx(
  actual::ITensor, expected::ITensor, msg=""; kwargs...
)
  ChainRulesTestUtils.@test_msg msg isapprox(actual, expected; kwargs...)
end

function ChainRulesTestUtils.test_approx(
  actual::ITensor, expected::Number, msg=""; kwargs...
)
  ChainRulesTestUtils.@test_msg msg isapprox(actual[], expected; kwargs...)
end

function ChainRulesTestUtils.test_approx(
  actual::Number, expected::ITensor, msg=""; kwargs...
)
  ChainRulesTestUtils.@test_msg msg isapprox(actual, expected[]; kwargs...)
end

@testset "ITensorChainRules.jl" begin
  @testset "Basic rrules" begin
    i = Index(2, "i")
    A = randomITensor(i', dag(i))
    Ac = randomITensor(ComplexF64, i', dag(i))
    B = randomITensor(i', dag(i))
    C = ITensor(3.4)

    test_rrule(getindex, ITensor(3.4); check_inferred=false)
    test_rrule(getindex, A, 1, 2; check_inferred=false)
    test_rrule(*, A', A; check_inferred=false)
    test_rrule(*, 3.2, A; check_inferred=false)
    test_rrule(*, A, 4.3; check_inferred=false)
    test_rrule(+, A, B; check_inferred=false)
    test_rrule(prime, A; check_inferred=false)
    test_rrule(prime, A, 2; check_inferred=false)
    test_rrule(prime, A; fkwargs=(; tags="i"), check_inferred=false)
    test_rrule(prime, A; fkwargs=(; tags="x"), check_inferred=false)
    test_rrule(replaceprime, A, 1 => 2; check_inferred=false)
    test_rrule(swapprime, A, 0 => 1; check_inferred=false)
    test_rrule(addtags, A, "x"; check_inferred=false)
    test_rrule(addtags, A, "x"; fkwargs=(; plev=1), check_inferred=false)
    test_rrule(removetags, A, "i"; check_inferred=false)
    test_rrule(replacetags, A, "i" => "j"; check_inferred=false)
    test_rrule(
      swaptags, randomITensor(Index(2, "i"), Index(2, "j")), "i" => "j"; check_inferred=false
    )
    test_rrule(replaceind, A, i' => sim(i); check_inferred=false)
    test_rrule(replaceinds, A, (i, i') => (sim(i), sim(i)); check_inferred=false)
    test_rrule(swapind, A, i', i; check_inferred=false)
    test_rrule(swapinds, A, (i',), (i,); check_inferred=false)
    test_rrule(itensor, randn(2, 2), i', i; check_inferred=false)
    test_rrule(ITensor, randn(2, 2), i', i; check_inferred=false)
    test_rrule(ITensor, 2.3; check_inferred=false)
    test_rrule(dag, A; check_inferred=false)
    test_rrule(permute, A, reverse(inds(A)); check_inferred=false)

    f = x -> sin(scalar(x)^3)
    args = (C,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> sin(x[]^3)
    args = (C,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = adjoint
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = (x, y) -> (x * y)[1, 1]
    args = (A', A)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> prime(x, 2)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> x'[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> addtags(x, "x")[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x' * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (prime(x) * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> ((x'' * x') * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x'' * (x' * x))[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = (x, y, z) -> (x * y * z)[1, 1]
    args = (A'', A', A)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x'' * x' * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x''' * x'' * x' * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x''' * x'' * x' * x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = (x, y) -> (x + y)[1, 1]
    args = (A, B)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x + x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (2x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x + 2x)[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x + 2 * mapprime(x' * x, 2 => 1))[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = (x, y) -> (x * y)[]
    args = (A, δ(dag(inds(A))))
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x * x)[]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x * δ(dag(inds(x))))[]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = function (x)
      y = x' * x
      tr = δ(dag(inds(y)))
      return (y * tr)[]
    end
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = function (x)
      y = x'' * x' * x
      tr = δ(dag(inds(y)))
      return (y * tr)[]
    end
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x^2 * δ((i', i)))[1, 1]
    args = (6.2,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> (x^2 * δ(i', i))[1, 1]
    args = (5.2,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> itensor([x^2 x; x^3 x^4], i', i)
    args = (2.54,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> ITensor([x^2 x; x^3 x^4], i', i)
    args = (2.1,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> ITensor(x)
    args = (2.12,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = function (x)
      j = Index(2)
      T = itensor([x^2 sin(x); x^2 exp(-2x)], j', dag(j))
      return real((dag(T) * T)[])
    end
    args = (2.8,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    args = (2.8 + 3.1im,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = function f(x)
      j = Index(2)
      v = itensor([exp(-3.2x), cos(2x^2)], j)
      T = itensor([x^2 sin(x); x^2 exp(-2x)], j', dag(j))
      return real((dag(v') * T * v)[])
    end
    args = (2.8,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    args = (2.8 + 3.1im,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = function (x)
      j = Index(2)
      return real((x^3 * ITensor([sin(x) exp(-2x); 3x^3 x+x^2], j', dag(j)))[1, 1])
    end
    args = (3.4 + 2.3im,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> prime(permute(x, reverse(inds(x))))[1, 1]
    args = (A,)
    test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    f = x -> prime(x; plev=1)[1, 1]
    args = (A,)
    @test_throws ErrorException f'(args...)
  end
  @testset "Energy minimization" begin
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
    E(ψ::ITensor) = E(H, ψ)
    ∇E(ψ::ITensor) = E'(ψ)
    fg(ψ::ITensor) = (E(ψ), ∇E(ψ))
    linesearch = HagerZhangLineSearch(; c₁=.1, c₂=.9, ϵ=1e-6, θ=1/2, γ=2/3, ρ=5., verbosity=0)
    algorithm = LBFGS(3; maxiter=30, gradtol=1e-8, linesearch=linesearch)
    ψ, fψ, gψ, numfg, normgradhistory = optimize(fg, ψ₀, algorithm)
    D, _ = eigen(H; ishermitian=true)
    @test E(H, ψ) < E(H, ψ₀)
    @test E(H, ψ) ≈ minimum(D)
  end
  @testset "Energy minimization (MPS)" begin
    N = 4
    χ = 4
    s = siteinds("S=1/2", N; conserve_qns=true)
    os = OpSum()
    for n in 1:(N - 1)
      os .+= 0.5, "S+", n, "S-", n+1
      os .+= 0.5, "S-", n, "S+", n+1
      os .+= "Sz", n, "Sz", n+1
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
    E(ψ) = E(H, ψ)
    ∇E(ψ) = E'(ψ)
    fg(ψ) = (E(ψ), ∇E(ψ))
    linesearch = HagerZhangLineSearch(; c₁=.1, c₂=.9, ϵ=1e-6, θ=1/2, γ=2/3, ρ=5.)
    algorithm = LBFGS(5; maxiter=100, gradtol=1e-8, linesearch=linesearch, verbosity=2)
    ψ, fψ, gψ, numfg, normgradhistory = optimize(fg, ψ₀, algorithm)
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, χ)
    fψmps, ψmps = dmrg(Hmpo, ψ₀mps, sweeps)
    time_Eψ = @elapsed E(ψ)
    time_∇Eψ = @elapsed E'(ψ)
    @test E(H, ψ) ≈ inner(ψmps, Hmpo, ψmps) / inner(ψmps, ψmps)
  end
end
