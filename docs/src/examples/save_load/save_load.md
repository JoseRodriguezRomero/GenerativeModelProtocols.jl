# Saving and Loading Models

This module provides convenient functionality for saving and loading trained 
generative models in the [HDF5](https://hdfgroup.org) file format. 
[HDF5](https://hdfgroup.org) was specifically chosen to maximize 
interoperability. This ensures your trained models remain accessible and usable 
across other software solutions, even if those systems do not use this module or 
the Julia programming language at all.

To access this functionality, one only needs to load the Julia modules 
[`FileIO.jl`](https://github.com/juliaio/fileio.jl) and
[`HDF5.jl`](https://github.com/JuliaIO/HDF5.jl)
```julia
using GenerativeModelProtocols
using FileIO, HDF5
```
finally, we can save an already trained model
```julia-repl
julia> protocol
GenerativeModelProtocol:
training_data      = 2×5000 Matrix{Float64}
mean_training_data = Tuple{Float32, Float64}
var_training_data  = Tuple{Float32, Float64}
model              = GenerativeModelProtocols.NormalizingFlow

julia> protocol.model
GenerativeModelProtocols.NormalizingFlow:
velocity_field: 
   Chain(
      Dense(3 => 32, swish)
      Dense(32 => 32, swish)
      Dense(32 => 32, swish)
      Dense(32 => 32, swish)
      Dense(32 => 32, swish)
      Dense(32 => 2)
   )
```
to a file named `file_name.h5` with a single function call
```julia-repl
julia> save("file_name.h5", protocol)
```

```@raw html
<div style="text-align: center; margin: 0px 0;">
    <img src="../../../assets/examples/save_load/hdfview_screenshot.png" alt="Description" style="max-width: 100% !important; width: auto !important; height: auto !important; display: inline-block; background: transparent !important; box-shadow: none !important; border: none !important;">
</div>
```

Likewise, we can load our saved model again with a single function call
```julia-repl
julia> loaded_protocol = GenerativeModelProtocol("file_name.h5")
GenerativeModelProtocol:
training_data      = 2×0 Matrix{Float64}
mean_training_data = Tuple{Float64, Float64}
var_training_data  = Tuple{Float64, Float64}
model              = GenerativeModelProtocols.NormalizingFlow

julia> loaded_protocol.model
GenerativeModelProtocols.NormalizingFlow:
velocity_field: 
   Chain(
      Dense(3 => 32, swish)
      Dense(32 => 32, swish)
      Dense(32 => 32, swish)
      Dense(32 => 32, swish)
      Dense(32 => 32, swish)
      Dense(32 => 2)
   )
```
It is worth pointing out that saved models/protocols are stripped of their 
training data. Thus, to continue traning a saved model, `train_data` must be 
supplied separately by the user with
```julia-repl
julia> loaded_protocol = GenerativeModelProtocol("file_name.h5", train_data)
GenerativeModelProtocol:
training_data      = 2×5000 Matrix{Float64}
mean_training_data = Tuple{Float64, Float64}
var_training_data  = Tuple{Float64, Float64}
model              = GenerativeModelProtocols.NormalizingFlow

julia> train!(loaded_protocol; epochs = 50);
Training Diffusion Model...
Epoch        1 | Mean MSE:   1.48582902e+00 
Epoch        5 | Mean MSE:   1.43085193e+00 
Epoch       10 | Mean MSE:   1.44327017e+00 
Epoch       15 | Mean MSE:   1.43267012e+00 
Epoch       20 | Mean MSE:   1.43097105e+00 
Epoch       25 | Mean MSE:   1.43716303e+00 
Epoch       30 | Mean MSE:   1.41456013e+00 
Epoch       35 | Mean MSE:   1.42213919e+00 
Epoch       40 | Mean MSE:   1.45697769e+00 
Epoch       45 | Mean MSE:   1.40971190e+00 
Epoch       50 | Mean MSE:   1.40970298e+00 
Training complete!
Training took 37 seconds.
```

# HDF5 Group Names

By default the main group name used for all saved `GenerativeModelProtocol` 
instances is `generative_model_protocol`, the default group name for the trained
parameters of its generative model is `generative_model` and the default group 
name for all its metadata is `metadata`. However, users can easily modify all or 
some of these names by simply using the corresponding key word arguments.

For example, we can save our example `protocol` with customized HDF5 group 
names as
```julia
save("saved_protocol.h5", protocol;
   main_group_name             = "customized_protocol",
   metadata_group_name         = "customized_metadata",
   generative_model_group_name = "customized_generative_model"
)
```

```@raw html
<div style="text-align: center; margin: 0px 0;">
    <img src="../../../assets/examples/save_load/hdfview_screenshot_customized.png" alt="Description" style="max-width: 100% !important; width: auto !important; height: auto !important; display: inline-block; background: transparent !important; box-shadow: none !important; border: none !important;">
</div>
```

# User Metadata

Users can easily include additional metadata to any saved instance of a 
`GenerativeModelProtocol` via the following keyword argument
```julia
A = 5.0
B = "32"
C = [1, 3]

save("saved_protocol.h5", protocol;
   main_group_name             = "customized_protocol",
   metadata_group_name         = "customized_metadata",
   generative_model_group_name = "customized_generative_model"
   metadata                    = (; A, B, C)
)
```
or equivalently
```julia
metadata = Dict(
   "A" => 5.0,
   "B" => "32",
   "C" => [1, 3]
)

save("saved_protocol.h5", protocol;
   main_group_name             = "customized_protocol",
   metadata_group_name         = "customized_metadata",
   generative_model_group_name = "customized_generative_model"
   metadata                    = metadata
)
```

```@raw html
<div style="text-align: center; margin: 0px 0;">
    <img src="../../../assets/examples/save_load/hdfview_screenshot_metadata.png" alt="Description" style="max-width: 100% !important; width: auto !important; height: auto !important; display: inline-block; background: transparent !important; box-shadow: none !important; border: none !important;">
</div>
```
