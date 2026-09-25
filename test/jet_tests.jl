using GenerativeModelProtocols
using FileIO, HDF5, DifferentialEquations

using JET

@testset "JET.jl" begin
    JET.test_package(GenerativeModelProtocols, target_modules = (GenerativeModelProtocols,))
end

