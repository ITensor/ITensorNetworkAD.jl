using ITensors
using ITensorNetworkAD
using ITensorNetworkAD.ITensorNetworks: Models

@testset "test local hamiltonian builder" begin
  Nx = 2
  Ny = 3
  sites = siteinds("S=1/2", Ny, Nx)
  H = Models.mpo(Models.Model("tfim"), sites; h=1.0)
  H_local = Models.localham(Models.Model("tfim"), sites; h=1.0)
  @test Models.checklocalham(H_local, H, sites)
end
