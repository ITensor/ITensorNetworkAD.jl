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

  @info get_index_groups([[[t1, t2], [t3]], [t4]])
  tn_ig_dict, ig_neighbor_set = index_group_info([[[t1, t2], [t3]], [t4]])
  @info tn_ig_dict
  @info ig_neighbor_set
end
