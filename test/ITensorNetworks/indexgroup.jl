using ITensors
using ITensorNetworkAD.ITensorNetworks: index_group_info, IndexGroup, get_index_groups

@testset "test index_group_info" begin
  a = Index(2, "a")
  b = Index(2, "b")
  c = Index(2, "c")
  d = Index(2, "d")
  t1 = ITensor(a, b)
  t2 = ITensor(b, c)
  t3 = ITensor(c, d)
  t4 = ITensor(a, d)
  tn = [[[t1, t2], [t3]], [t4]]
  index_groups = get_index_groups(tn)
  tn_ig_dict = Dict{Vector,Vector{IndexGroup}}()
  for c in get_leaves(tn)
    tn_ig_dict[c] = neighboring_index_groups(c, index_groups)
  end
  tn_ig_dict, ig_neighbor_set = index_group_info(tn)
  @assert length(ig_neighbor_set) == 1
end
