using NearestNeighbors
using LinearAlgebra
using StaticArrays

# function is_in_fov(x::SVector{3,F}, pt::SVector{2,F}, fovRadius::F, fovAngle::F) where {F}
#     return 
# end

function is_in_fov(x::SVector{3,F}, pt::SVector{2,F}, fovRadius::F, fovAngle::F, robustness::F = zero(F)) where {F}
    # 1. Calculate the Apex Shift
    # The "safe" cone starts further ahead.
    # d_shift = maxError / sin(halfAngle)
    shift = robustness / sin(fovAngle / 2)
    
    # 3. Calculate Reduced Radius
    # We must reduce the radius by the uncertainty (maxError) AND the shift distance
    # to be conservative (ensuring we don't exceed the original outer arc).
    robustRadius = fovRadius - robustness - shift
    
    if robustRadius <= zero(F)
        return false
    end
    # 2. Create the Virtual Pose
    # Move the robot pose forward along its heading by 'shift'
    theta = x[3]
    s, c = sincos(theta)
    
    # New position: x + shift * cos(theta), y + shift * sin(theta)
    shift_vec = SVector(c * shift, s * shift)
    new_pos_2d = x[SOneTo(2)] + shift_vec
    
    # Reassemble into 3D state (x, y, theta)
    new_x = SVector(new_pos_2d[1], new_pos_2d[2])
    
    # 4. Delegate to standard check
    return norm(pt - new_x) < robustRadius && abs(angle_diff(atan(pt[2]-new_x[2], pt[1]-new_x[1]), theta)) < fovAngle/2
end

function features_in_fov_idx(features::BallTree, x::SVector{3,F}, fovRadius::F, fovAngle::F, robustness::F=zero(F)) where {F}
    features_in_radius = Int16[]
    sizehint!(features_in_radius, length(features.data))
    shift = robustness / sin(fovAngle / 2)
    # create shifted position
    x_shifted = x[SOneTo(2)] + SVector(cos(x[3]), sin(x[3])) * shift
    features_in_fov = Int16[]
    newRadius = fovRadius - robustness - shift
    if newRadius <= zero(F)
        return features_in_fov
    end
    inrange!(features_in_radius, features, x_shifted, newRadius)
    # check if features are within the fov fovAngle
    yaw_vec = @SVector[cos(x[3]), sin(x[3])]
    sizehint!(features_in_fov, length(features_in_radius))
    for i in features_in_radius
        pt = features.data[i] - x_shifted
        pt = pt/norm(pt)
        if pt⋅yaw_vec >= cos(fovAngle/2)
            push!(features_in_fov, i)
        end
    end

    return features_in_fov
end

# function features_in_fov_idx_robust(features::BallTree, x::SVector{3,F}, fovRadius::F, fovAngle::F, maxError::F) where {F}
#     return features_in_fov_idx(features, x_shifted, fovRadius - maxError - (maxError / sin(fovAngle / 2)), fovAngle)
# end

