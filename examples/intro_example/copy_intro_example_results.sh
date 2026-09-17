#!bin/bash

for type in dm gan gmm nf vae
do
    cp "${type}_example"/*.svg "../../docs/src/assets/examples/getting_started/${type}/"
done

