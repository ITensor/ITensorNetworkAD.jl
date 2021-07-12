using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using ITensors
using ITensors.NDTensors
using ITensorNetworkAD
using Random
using Test
using Zygote

using Zygote: ZygoteRuleConfig

#
# ITensor extensions
#

# TODO: this is to fix an error within `ChainRulesTestUtils._test_add!!_behaviour`
# that gets called in `test_rrule`. Is this needed/a good definition?
Base.:+(x::Number, y::ITensor) = ITensor(x) + y
Base.:+(x::ITensor, y::Number) = x + ITensor(y)

#
# For ITensor compatibility with FiniteDifferences
#

function FiniteDifferences.to_vec(A::ITensor)
  # TODO: generalize to sparse tensors
  return vec(array(A)), x -> itensor(x, inds(A))
end
function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::ITensor)
  # TODO: generalize to sparse tensors
  return randomITensor(eltype(A), inds(A))
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::ITensor)
  # TODO: generalize to sparse tensors
  return randomITensor(eltype(A), inds(A))
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::Tensor)
  # TODO: generalize to sparse tensors
  return randomTensor(eltype(A), inds(A))
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, x::Index)
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
  i = Index(2)
  A = randomITensor(i', dag(i))
  B = randomITensor(i', dag(i))

  test_rrule(getindex, A, 1, 1; check_inferred=false)
  test_rrule(getindex, A, 1, 2; check_inferred=false)
  test_rrule(*, A', A; check_inferred=false)
  test_rrule(*, 3.2, A; check_inferred=false)
  test_rrule(*, A, 4.3; check_inferred=false)
  test_rrule(+, A, B; check_inferred=false)
  test_rrule(prime, A; check_inferred=false)
  test_rrule(prime, A, 2; check_inferred=false)
  test_rrule(addtags, A, "i"; check_inferred=false)
  test_rrule(settags, A, "x,y"; check_inferred=false)
  # XXX: broken with some ambiguity error in ChainRulesTestUtils
  #test_rrule(delta, (i', i); check_inferred=false)
  test_rrule(itensor, randn(2, 2), i', i; check_inferred=false)
  test_rrule(ITensor, randn(2, 2), i', i; check_inferred=false)
  test_rrule(ITensor, 2.3; check_inferred=false)

  f = adjoint
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = (x, y) -> (x * y)[1, 1]
  args = (A', A)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> prime(x, 2)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> x'[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> addtags(x, "x")[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x' * x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (prime(x) * x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> ((x'' * x') * x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x'' * (x' * x))[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = (x, y, z) -> (x * y * z)[1, 1]
  args = (A'', A', A)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x'' * x' * x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x''' * x'' * x' * x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x''' * x'' * x' * x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = (x, y) -> (x + y)[1, 1]
  args = (A, B)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x + x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (2x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x + 2x)[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x + 2 * mapprime(x' * x, 2 => 1))[1, 1]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = (x, y) -> (x * y)[]
  args = (A, δ(dag(inds(A))))
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x * x)[]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x * δ(dag(inds(x))))[]
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = function (x)
    y = x' * x
    tr = δ(dag(inds(y)))
    return (y * tr)[]
  end
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = function (x)
    y = x'' * x' * x
    tr = δ(dag(inds(y)))
    return (y * tr)[]
  end
  args = (A,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x^2 * δ((i', i)))[1, 1]
  args = (2.2,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> (x^2 * δ(i', i))[1, 1]
  args = (2.2,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> itensor([x^2 x; x^3 x^4], i', i)
  args = (2.3,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> ITensor([x^2 x; x^3 x^4], i', i)
  args = (2.3,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
  f = x -> ITensor(x)
  args = (2.3,)
  test_rrule(ZygoteRuleConfig(), f, args..., rrule_f=rrule_via_ad, check_inferred=false)
end
