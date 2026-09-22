using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
    GenerativeModelProtocols;
    ambiguities=(exclude=[], broken=false),
    stale_deps=(ignore=Symbol[],),
    deps_compat=(ignore=Symbol[],),
    piracies=true,
  )
end

