using ChainRulesCore

function ChainRulesCore.rrule(
  ::typeof(split_network),
  tn::Matrix{ITensor};
  projector_center=default_projector_center(tn),
  rotation=false,
)
  function pullback(dtn_split::Matrix{ITensor})
    dtn = map(t -> replaceprime(t, 1 => 0), dtn_split)
    return (NoTangent(), dtn)
  end
  return split_network(tn; projector_center=projector_center, rotation=rotation), pullback
end
