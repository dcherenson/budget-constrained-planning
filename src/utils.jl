using Dubins
using StaticArrays

function wrapToPi(x)
    return atan(sin(x), cos(x))
end

function angle_diff(a, b)
    return wrapToPi(a - b)
end

function angle_add(a, b)
    return wrapToPi(a + b)
end

"""
    dubins_dynamics(x::SVector{3,F}, u::SVector{2,F}, dt::F, turning_radius::F) where F

Propagate Dubins car dynamics forward by dt using control input u = [v, ω].
Returns new state [x, y, θ].
"""
function dubins_dynamics(x::SVector{3,F}, u::SVector{2,F}, dt::F, turning_radius::F) where F
    v, ω = u
    θ = x[3]
    
    # Dubins dynamics: ẋ = v cos(θ), ẏ = v sin(θ), θ̇ = ω
    # Clamp angular velocity based on turning radius and velocity
    max_ω = v / turning_radius
    ω = clamp(ω, -max_ω, max_ω)
    
    # Simple Euler integration
    x_new = x[1] + v * cos(θ) * dt
    y_new = x[2] + v * sin(θ) * dt
    θ_new = angle_add(θ, ω * dt)
    
    return SVector(x_new, y_new, θ_new)
end

"""
    update_state_estimate(x_est::SVector{3,F}, x_true::SVector{3,F}, u::SVector{2,F}, 
                         dt::F, error_rate::F) where F

Update state estimate using odometry with error accumulation.
Position drifts based on error_rate, yaw is always known from true state.
Returns new estimate [x_est, y_est, θ_true].
"""
function update_state_estimate(x_est::SVector{3,F}, x_true::SVector{3,F}, u::SVector{2,F}, 
                               dt::F, error_rate::F) where F
    v, ω = u
    θ = x_true[3]  # Yaw is always known
    
    # Integrate position using estimated state heading
    # Add error proportional to distance traveled
    distance_traveled = v * dt
    error_magnitude = error_rate * distance_traveled
    
    # Random error direction (in practice this would be more sophisticated)
    # For simulation, we'll use a simple drift model
    x_new = x_est[1] + v * cos(θ) * dt + error_magnitude * cos(θ + π/4)
    y_new = x_est[2] + v * sin(θ) * dt + error_magnitude * sin(θ + π/4)
    
    return SVector(x_new, y_new, θ)
end

"""
    reset_state_estimate(x_true::SVector{3,F}) where F

Reset state estimate to true state (e.g., when landmark is observed).
"""
function reset_state_estimate(x_true::SVector{3,F}) where F
    return x_true
end

"""
    waypoint_controller(x::SVector{3,F}, target::SVector{3,F}, max_velocity::F, 
                       turning_radius::F, lookahead_dist::F=5.0) where F

Simple waypoint tracking controller that computes control input [v, ω] to track a target waypoint.
Uses pure pursuit style control with heading feedback.
"""
function waypoint_controller(x::SVector{3,F}, target::SVector{3,F}, max_velocity::F, 
                            turning_radius::F, lookahead_dist::F=5.0) where F
    # Compute distance and angle to target
    dx = target[1] - x[1]
    dy = target[2] - x[2]
    dist = sqrt(dx^2 + dy^2)
    
    # Desired heading to target
    θ_desired = atan(dy, dx)
    
    # Heading error
    θ_error = angle_diff(θ_desired, x[3])
    
    # Velocity is constant at max
    v = max_velocity
    
    # Angular velocity using proportional control
    # Higher gain when far from target, lower when close
    k_angular = 2.0  # proportional gain
    ω = k_angular * θ_error
    
    # If very close to target, also try to match target heading
    if dist < lookahead_dist
        heading_error = angle_diff(target[3], x[3])
        ω = k_angular * θ_error + 0.5 * heading_error
    end
    
    # Clamp angular velocity
    max_ω = v / turning_radius
    ω = clamp(ω, -max_ω, max_ω)
    
    return SVector(v, ω)
end

"""
    get_trajectory_target(traj::CompositeTrajectory, t_rel::F, vel::F, lookahead::F) where F

Get target waypoint from trajectory at current time + lookahead distance.
"""
function get_trajectory_target(traj, t_rel::F, vel::F, lookahead::F) where F
    # Sample trajectory at lookahead distance
    s_current = t_rel * vel
    s_target = s_current + lookahead
    
    # Get total trajectory length
    total_length = 0.0
    if t_rel < traj.switch_time
        for p in traj.nominal_trajectory
            total_length += dubins_path_length(p)
        end
        for p in traj.backup_trajectory
            total_length += dubins_path_length(p)
        end
        
        if s_target > total_length
            s_target = total_length
        end
        return sample_committed_trajectory(traj, s_target / vel, vel)
    else
        # In backup trajectory
        t_backup = t_rel - traj.switch_time
        for p in traj.backup_trajectory
            total_length += dubins_path_length(p)
        end
        
        s_backup = t_backup * vel
        s_target_backup = s_backup + lookahead
        
        if s_target_backup > total_length
            s_target_backup = total_length
        end
        
        return sample_combined_dubins_path(traj.backup_trajectory, s_target_backup)
    end
end

function make_path_from_waypoints(waypoints::Vector{SVector{3,F}}, r::F) where {F}
    # construct the best path
    path = DubinsPath[]
    sizehint!(path, length(waypoints)-1)
    for i=1:length(waypoints)-1
        e, p = dubins_shortest_path(waypoints[i], waypoints[i+1], r, 1e-3)
        @assert e == Dubins.EDUBOK
        push!(path, p)
    end
    return path
end

function sample_combined_dubins_path(path::Vector{DubinsPath}, s::F) where {F}
    # sample the combined path
    s_init = s
    for p in path
        L = dubins_path_length(p)
        if s < L
            e, q = dubins_path_sample(p, s)
            @assert e == Dubins.EDUBOK
            return q
        end
        s -= L
    end
    e,q = dubins_path_endpoint(path[end])
    @assert e == Dubins.EDUBOK
    return q
end

function dubins_sub_trajectory(path::Vector{DubinsPath}, s::F) where {F}
    # sample the combined path
    sub_path = DubinsPath[]
    for p in path
        L = dubins_path_length(p)
        if s < L
            e, q = dubins_extract_subpath(p, s)
            @assert e == Dubins.EDUBOK
            push!(sub_path, q)
            return sub_path
        else
            push!(sub_path, p)
            s -= L
        end
    end

    return sub_path
end

function dubins_traj_length(path::Vector{DubinsPath})
    L = 0.0
    for p in path
        L += dubins_path_length(p)
    end
    return L
end