
# Needed for defining the rule for `adjoint(A::ITensor)`
# which currently doesn't work by overloading `ChainRulesCore.rrule`
using ZygoteRules: @adjoint

@adjoint function Base.adjoint(x::ITensor)
  y = prime(x)
  function setinds_pullback(ȳ)
    x̄ = ITensors.setinds(ȳ, inds(x))
    return (x̄,)
  end
  return y, setinds_pullback
end
