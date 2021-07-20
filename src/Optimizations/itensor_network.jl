using ChainRulesCore
using ..ITensorNetworks
using ..ITensorNetworks: split_links

inv_op(::typeof(addtags)) = :removetags
inv_op(::typeof(removetags)) = :addtags

function ChainRulesCore.rrule(
  ::typeof(split_links),
  H::Union{MPS,MPO};
  split_tags=("" => ""),
  split_plevs=(0 => 1),
  tag_f=addtags,
)
  function pullback(dHsplit)
    dH = split_links(
      dHsplit;
      split_tags=split_tags,
      split_plevs=(split_plevs[2] => split_plevs[1]),
      tag_f=inv_op(tag_f),
    )
    return (NoTangent(), dH, NoTangent(), NoTangent(), NoTangent())
  end
  return split_links(H; split_tags=split_tags, split_plevs=split_plevs, tag_f=tag_f),
  pullback
end
