@reexport module ITensorChainRules

using ChainRulesCore
using ITensors

# TODO: move to ITensors.jl.
# Useful for generic code.
ITensors.dag(n::Number) = conj(n)

# Needed for defining the rule for `adjoint(A::ITensor)`
# which currently doesn't work by overloading `ChainRulesCore.rrule`
using ZygoteRules: @adjoint

function ChainRulesCore.rrule(::typeof(getindex), x::ITensor, I...)
  y = getindex(x, I...)
  function getindex_pullback(ȳ)
    # TODO: add definition `ITensor(::Tuple{}) = ITensor()`
    # to ITensors.jl so no splatting is needed here.
    x̄ = ITensor(inds(x)...)
    x̄[I...] = ȳ
    Ī = broadcast(_ -> NoTangent(), I)
    return (NoTangent(), x̄, Ī...)
  end
  return y, getindex_pullback
end

# Specialized version in order to avoid call to `setindex!`
# within the pullback, should be better for taking higher order
# derivatives in Zygote.
function ChainRulesCore.rrule(::typeof(getindex), x::ITensor)
  y = x[]
  function getindex_pullback(ȳ)
    x̄ = ITensor(ȳ)
    return (NoTangent(), x̄)
  end
  return y, getindex_pullback
end

function setinds_pullback(ȳ, x, a...)
  x̄ = ITensors.setinds(ȳ, inds(x))
  ā = broadcast(_ -> NoTangent(), a)
  return (NoTangent(), x̄, ā...)
end

for fname in (
  :prime,
  :setprime,
  :noprime,
  :replaceprime,
  :swapprime,
  :addtags,
  :removetags,
  :replacetags,
  :settags,
  :swaptags,
  :replaceind,
  :replaceinds,
  :swapind,
  :swapinds,
)
  @eval begin
    function ChainRulesCore.rrule(::typeof($fname), x::ITensor, a...)
      y = $fname(x, a...)
      function f_pullback(ȳ)
        return setinds_pullback(ȳ, x, a...)
      end
      return y, f_pullback
    end
  end
end

# TODO: This is not being called by Zygote for some reason,
# using a Zygote overload directly instead. Figure out
# why, maybe raise an issue.
#function ChainRulesCore.rrule(::typeof(adjoint), x::ITensor)
#  y = prime(x)
#  function adjoint_pullback(ȳ)
#    return setinds_pullback(ȳ, x)
#  end
#  return y, adjoint_pullback
#end

@adjoint function Base.adjoint(x::ITensor)
  y = prime(x)
  function setinds_pullback(ȳ)
    x̄ = ITensors.setinds(ȳ, inds(x))
    return (x̄,)
  end
  return y, setinds_pullback
end

function _contract_pullback(ȳ, x1, x2)
  x̄1 = ȳ * dag(x2)
  x̄2 = dag(x1) * ȳ
  return (NoTangent(), x̄1, x̄2)
end

function _rrule(::typeof(*), x1, x2)
  y = x1 * x2
  function contract_pullback(ȳ)
    return _contract_pullback(ȳ, x1, x2)
  end
  return y, contract_pullback
end

# Special case for contracting a pair of ITensors
function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::ITensor)
  return _rrule(*, x1, x2)
end

function ChainRulesCore.rrule(::typeof(*), x1::Number, x2::ITensor)
  return _rrule(*, x1, x2)
end

function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::Number)
  return _rrule(*, x1, x2)
end

function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::ITensor, xs::ITensor...)
  y = *(x1, x2, xs...)
  function contract_pullback(ȳ)
    # TODO: use some contraction sequence optimization here
    tn = [x1, x2, xs...]
    N = length(tn)
    env_contracted = Vector{ITensor}(undef, N)
    for n in 1:length(tn)
      tn_left = tn[1:(n - 1)]
      # TODO: define contract([]) = ITensor(1.0)
      env_left = isempty(tn_left) ? ITensor(1.0) : contract(tn_left)
      tn_right = tn[reverse((n + 1):end)]
      env_right = isempty(tn_right) ? ITensor(1.0) : contract(tn_right)
      env_contracted[n] = dag(env_left) * ȳ * dag(env_right)
    end
    return (NoTangent(), env_contracted...)
  end
  return y, contract_pullback
end

function ChainRulesCore.rrule(::typeof(+), x1::ITensor, x2::ITensor)
  y = x1 + x2
  function add_pullback(ȳ)
    return (NoTangent(), ȳ, ȳ)
  end
  return y, add_pullback
end

function ChainRulesCore.rrule(::typeof(itensor), x::Array, a...)
  y = itensor(x, a...)
  function itensor_pullback(ȳ)
    x̄ = array(ȳ)
    ā = broadcast(_ -> NoTangent(), a)
    return (NoTangent(), x̄, ā...)
  end
  return y, itensor_pullback
end

function ChainRulesCore.rrule(::typeof(ITensor), x::Array, a...)
  y = ITensor(x, a...)
  function ITensor_pullback(ȳ)
    # TODO: define `Array(::ITensor)` directly
    x̄ = Array(ȳ, inds(ȳ)...)
    ā = broadcast(_ -> NoTangent(), a)
    return (NoTangent(), x̄, ā...)
  end
  return y, ITensor_pullback
end

function ChainRulesCore.rrule(::typeof(ITensor), x::Number)
  y = ITensor(x)
  function ITensor_pullback(ȳ)
    x̄ = ȳ[]
    return (NoTangent(), x̄)
  end
  return y, ITensor_pullback
end

function ChainRulesCore.rrule(::typeof(dag), x::ITensor)
  y = dag(x)
  function dag_pullback(ȳ)
    x̄ = dag(ȳ)
    return (NoTangent(), x̄)
  end
  return y, dag_pullback
end

@non_differentiable Index(::Any...)
@non_differentiable delta(::Any...)
@non_differentiable dag(::Index)
@non_differentiable inds(::Any...)

end
