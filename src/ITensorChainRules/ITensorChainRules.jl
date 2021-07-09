@reexport module ITensorChainRules

using ChainRulesCore
using ITensors

function ChainRulesCore.rrule(::typeof(getindex), x::ITensor, I...)
  y = getindex(x, I...)
  function getindex_pullback(ȳ)
    x̄ = ITensor(inds(x))
    x̄[I...] = ȳ
    Ī = broadcast(_ -> NoTangent(), I)
    return (NoTangent(), x̄, Ī...)
  end
  return y, getindex_pullback
end

function ChainRulesCore.rrule(::typeof(*), x1::ITensor, x2::ITensor)
  y = x1 * x2
  function contract_pullback(ȳ)
    x̄1 = ȳ * x2
    x̄2 = x1 * ȳ
    return (NoTangent(), x̄1, x̄2)
  end
  return y, contract_pullback
end

end
