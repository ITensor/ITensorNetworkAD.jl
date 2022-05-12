using ITensorNetworkAD.ITensorNetworks:
  merge_tree, subtree, vectorize, find_topo_sort, get_leaves

@testset "test merge tree" begin
  t1 = [[1], [2], [3]]
  t2 = [4, 5, 6]
  @assert merge_tree(t1, t2; append=true) == [[1], [2], [3], [4, 5, 6]]
  @assert merge_tree(t1, t2; append=false) == [[[1], [2], [3]], [4, 5, 6]]
  @assert merge_tree([], [1, 2, 3]; append=false) == [1, 2, 3]
end

@testset "test subtree and vectorize" begin
  t1 = [[[1, 2], [3]], [4]]
  subset = [1]
  @assert subtree(t1, subset) == [1]
  @assert vectorize(t1) == [1, 2, 3, 4]
end

@testset "test find topo sort" begin
  tn = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  @assert length(find_topo_sort(tn)) == 7
  @assert length(find_topo_sort(tn, get_leaves(tn))) == 3
end
