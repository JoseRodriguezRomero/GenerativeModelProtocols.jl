#!bin/bash

for dir in dm_example gan_example gmm_example nf_example vae_example
do
    cd "$dir"
    julia --threads 16 intro_example.jl
    cd ..
done

