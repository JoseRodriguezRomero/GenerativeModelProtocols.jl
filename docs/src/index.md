# GenerativeModelProtocols.jl

A unified Julia framework for building, training, and deploying generative 
models. 

This library wraps diverse architectures into a consistent 
`AbstractGenerativeModel` interface. Built on top of 
[`Lux.jl`](https://lux.csail.mit.edu/stable/), it delivers a developer-friendly 
experience with native GPU acceleration and maintainable code.

## Key Features

* **Unified Interface**: Standardized protocol for all generative architectures.
* **Lux Ecosystem**: Seamless integration with modern Lux.jl neural networks.
* **Hardware Accelerated**: Out-of-the-box support for execution on GPUs.
* **Model Persistence**: Built-in serialization mechanisms to save and 
resume workflows.

## Storage & Serialization

The package supports saving and loading trained model states for sharing and 
reuse. 

To keep the core package lightweight, serialization is handled via package 
extensions. Loading [`FileIO.jl`](https://github.com/juliaio/fileio.jl) and 
[`HDF5.jl`](https://github.com/JuliaIO/HDF5.jl) in your environment 
automatically enables the [HDF5](https://hdfgroup.org) storage backend.

## Supported Architectures

While the library includes built-in implementations of popular generative 
models, it is built for extensibility. You can easily integrate custom 
architectures by subtyping the `AbstractGenerativeModel` interface and 
implementing the required protocol methods.

The currently built-in generative model architectures are:

* (Hierarchical) [Variational Autoencoders](https://arxiv.org/abs/1312.6114)
* [Gaussian Mixture Models](https://link.springer.com/rwe/10.1007/978-0-387-73003-5_196)
* [Diffusion Models](https://arxiv.org/abs/1503.03585)
* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
* [Normalizing Flows](https://arxiv.org/abs/1505.05770)
