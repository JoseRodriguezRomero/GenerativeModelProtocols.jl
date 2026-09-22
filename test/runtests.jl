using Test
using ParallelTestRunner

using GenerativeModelProtocols

testsuite = ParallelTestRunner.find_tests(@__DIR__)
delete!(testsuite, "utility_utils/tests_base_utils")

ParallelTestRunner.runtests(
    GenerativeModelProtocols, 
    Base.ARGS; 
    testsuite = testsuite
)

