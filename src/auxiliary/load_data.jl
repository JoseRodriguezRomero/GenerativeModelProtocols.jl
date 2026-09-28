function load_data(data::Union{AbstractMatrix, Nothing}, batchsize::Int, shuffle::Bool = true, parallel::Bool = true)
    data = shuffle ? shuffleobs(data) : data

    if isnothing(data) || isempty(data)
        throw(ArgumentError("No training data was loaded!"))
    end

    return MLUtils.DataLoader(
        data, 
        batchsize = batchsize, 
        shuffle = false,
        parallel = parallel
    )
end

function load_data(data::Tuple{Vararg{Union{AbstractMatrix, Nothing}}}, batchsize::Int, shuffle::Bool = true, parallel::Bool = true)
    data = shuffle ? shuffleobs(data) : data

    if isnothing(data[1]) || isempty(data[1])
        throw(ArgumentError("No training data was loaded!"))
    end

    return DataLoader(
        data, 
        batchsize = batchsize, 
        shuffle = false,
        parallel = parallel
    )
end

