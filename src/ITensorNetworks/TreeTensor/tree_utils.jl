
# merge two trees
# new tree:
#    s
#   / \ 
# t1   t2
function merge_tree(t1::Vector, t2::Vector; append=false)
  if t2 == []
    return t1
  end
  if t1 == []
    return t2
  end
  if isleaf(t1) && isleaf(t2)
    return [t1, t2]
  end
  if isleaf(t1)
    return append ? [t1, t2...] : [t1, t2]
  end
  if isleaf(t2)
    return append ? [t1..., t2] : [t1, t2]
  end
  return append ? [t1..., t2...] : [t1, t2]
end

function isleaf(tree::Vector)
  if tree == []
    @info "tree is empty"
    return false
  end
  if all(v -> !(v isa Vector), tree)
    return true
  end
  return false
end

# get the subtree of tree that is in the subset
# example:
# subtree([[1, 2], [3, 4]], [1, 3]) = ([[1], [3]])
function subtree(tree::Vector, subset::Union{Vector,Tuple})
  if tree == []
    return []
  end
  if isleaf(tree)
    return intersect(tree, subset)
  end
  tree = [subtree(i, subset) for i in tree]
  tree = filter(t -> t != [], tree)
  if length(tree) == 1 && tree[1] isa Vector
    return tree[1]
  end
  return tree
end

# vectorize a tree
# example: [[1,2], [3,4]] = [1, 2, 3, 4]
function vectorize(tree)
  @assert tree != []
  if !(tree isa Vector)
    return [tree]
  end
  return mapreduce(vectorize, vcat, tree)
end
