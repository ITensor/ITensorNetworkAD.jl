using ITensors, ITensorNetworkAD
using ITensorNetworkAD.ITensorNetworks: TensorNetworkGraph, visualize, inds_network

N = (8, 8) #(12, 12)
linkdim = 2
cutoff = 1e-15
tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
tn = map(inds -> randomITensor(inds...), tn_inds)
tn = vec(tn[:, 1:4])
tng = TensorNetworkGraph(tn)
plt = visualize(tng)
plt
