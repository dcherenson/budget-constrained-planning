module VO

using LinearAlgebra
using NearestNeighbors
using StaticArrays
include("fov.jl")

@kwdef struct VOParams{F}
    fovRadius::F
    fovAngle::F
    errorRate::F
    updateRate::F
    minFeatures::Int
    maxError::F
end

function intersect_fast(a::Vector{I},b::Vector{I}) where {I <: Integer}
    c = Vector{I}()
    sizehint!(c, min(length(a), length(b)))
    if length(a) < length(b)
        for i = 1:length(a)
            if a[i] in b
                push!(c,a[i])
            end
        end
    else
        for i = 1:length(b)
            if b[i] in a
                push!(c,b[i])
            end
        end
    end
    return c
end


function odometry_error(start_pose::SVector{3,F}, end_pose::SVector{3,F}, features::BallTree, params::VOParams, robustness::F) where {F}
    # compute overlapping points
    startPoints = features_in_fov_idx(features, start_pose, params.fovRadius, params.fovAngle, robustness)    
    endPoints = features_in_fov_idx(features, end_pose, params.fovRadius, params.fovAngle, robustness)
    overlappingPoints = intersect_fast(startPoints, endPoints)

    if length(overlappingPoints) >= params.minFeatures
        cost = norm(start_pose[SOneTo(2)] - end_pose[SOneTo(2)])*params.errorRate
    else
        cost = Inf
    end

    return cost, length(overlappingPoints)
end

end