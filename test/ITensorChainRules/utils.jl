using ChainRulesTestUtils
using FiniteDifferences
using ITensors
using Random

using ChainRulesCore: NoTangent

# TODO: maybe move this into ITensorChainRules module?
# These are useful definitions for testing code. The downside
# is adding dependencies on FiniteDifferences and ChainRulesTestUtils

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
