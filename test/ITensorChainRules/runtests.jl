using ChainRulesTestUtils
using FiniteDifferences
using ITensors
using ITensorNetworkAD
using Random
using Test
using Zygote

using Zygote: ZygoteRuleConfig

# For ITensor compatibility with FiniteDifferences
function FiniteDifferences.to_vec(A::ITensor)
  # TODO: generalize to sparse tensors
  return vec(array(A)), x -> itensor(x, inds(A))
end
function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::ITensor)
  # TODO: generalize to sparse tensors
  return randomITensor(eltype(A), inds(A))
end

# For ITensor compatibility with ChainRulesTestUtils
function ChainRulesTestUtils.test_approx(actual::ITensor, expected::ITensor, msg=""; kwargs...)
  ChainRulesTestUtils.@test_msg msg isapprox(actual, expected; kwargs...)
end

@testset "ITensorChainRules.jl" begin
  i = Index(2)
  A = randomITensor(i', i)

  test_rrule(getindex, A, 1, 1; check_inferred=false)
  test_rrule(getindex, A, 1, 2; check_inferred=false)
  test_rrule(*, A', A; check_inferred=false)

  test_rrule(ZygoteRuleConfig(), (x, y) -> (x * y)[1, 1], A', A; rrule_f=rrule_via_ad, check_inferred=false)
end

