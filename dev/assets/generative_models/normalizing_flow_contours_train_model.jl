using Optimisers, Lux
using DifferentialEquations
using GenerativeModelProtocols

using FileIO, HDF5

function make_samples(num_samples)
    function sample_t0()
        x = randn(Float64)
        y = randn(Float64)

        return [x, y]
    end

    function sample_t1()
        p = rand()

        local x, y

        if p < 1.25 / 3.0
            x = 0.5 * randn(Float64) + 1.0
            y = 1.0 * randn(Float64) - 0.5
        else
            x = 1.0 * randn(Float64) - 3.0
            y = 0.6 * randn(Float64) + 1.0
        end

        return [x, y]
    end

    t0_samples = reduce(hcat, [sample_t0() for _ in 1:num_samples])
    t1_samples = reduce(hcat, [sample_t1() for _ in 1:num_samples])

    return t0_samples, t1_samples
end

_, t1_samples_train = make_samples(5000)

model = GenerativeModelProtocols.NormalizingFlow(2)
protocol = GenerativeModelProtocol(model, t1_samples_train;
    batchsize      = 256,
    epochs         = 3500,
    optimiser      = Adam(; eta = 1.0E-4, beta = (0.95, 0.999)),
    device         = cpu_device(),
    normalize_data = false
)
train!(protocol)
save("sample_nf_model.h5", protocol)

