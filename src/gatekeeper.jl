# include("world.jl")
using Dubins
using .World
using NearestNeighbors
using StaticArrays

@kwdef struct GatekeeperParams{F}
    ΔT::F = 0.1
    max_cost::F = 10.0
    propagation_dt::F = 0.1
    propagation_T_max::F = 10.0
end

@kwdef struct Gatekeeper
    params::GatekeeperParams = GatekeeperParams()
end

struct CompositeTrajectory{TN, TB, F}
    nominal_trajectory::TN
    backup_trajectory::TB 
    switch_time::F
    backup_time::F
    total_cost::F
end

function initialize_gatekeeper()
    gk = Gatekeeper()
    return gk
end

# returns true if new committed trajectory was created
function gatekeeper(x0::SVector{3,F}, committed_traj::CompositeTrajectory, nominal_planner::NominalPlanner, backup_planner::BackupPlanner, max_cost::F) where {F}
    new_committed = false
    
    # get the nominal trajectory
    nom_success, nom_path = query_nominal_planner(nominal_planner, x0)
    if !nom_success
        return new_committed, committed_traj
    end
    # convert it into a vector of dubinspaths
    nom_path = make_path_from_waypoints(nom_path, nominal_planner.prob.turning_radius)
    # initial switching time
    # form vector of cost over time for nominal trajectory
    nominal_cost_vs_time = compute_nominal_trajectory_cost(backup_planner.prob, nom_path, max_cost)
    T_S = length(nominal_cost_vs_time) / backup_planner.prob.vo_params.updateRate

    # search for T_S
    while T_S >= 0.0
        step_index = ceil(Int, T_S * backup_planner.prob.vo_params.updateRate)+1
        nom_cost = step_index > length(nominal_cost_vs_time) ? nominal_cost_vs_time[end] : nominal_cost_vs_time[step_index]
        if nom_cost < max_cost
            x_S = sample_combined_dubins_path(nom_path, T_S*backup_planner.prob.max_velocity)
            backup_cost, backup_path = query_backup_planner(backup_planner, x_S)
            if backup_cost + nom_cost < max_cost
                new_committed = true
                backup_traj = make_path_from_waypoints(backup_path, backup_planner.prob.turning_radius)
                committed_traj = CompositeTrajectory(dubins_sub_trajectory(nom_path, T_S*backup_planner.prob.max_velocity),
                                backup_traj,
                                T_S,
                                dubins_traj_length(backup_traj)/backup_planner.prob.max_velocity,
                                backup_cost + nom_cost)
                break
            end
        end
        T_S -= 1/backup_planner.prob.vo_params.updateRate
    end
    return new_committed, committed_traj

end

function compute_nominal_trajectory_cost(problem::BackupPlannerProblem, path::Vector{DubinsPath}, max_cost::F) where {F}
    cost_vector = [0.0]   
    step_size = problem.max_velocity / problem.vo_params.updateRate
    start_step = sample_combined_dubins_path(path, 0.0)
    s = step_size
    L = 0.0
    for p in path
        L += dubins_path_length(p)
    end

    while cost_vector[end] < max_cost && s < L
        end_step = sample_combined_dubins_path(path, s)
        landmark_in_fov = false
        for l in problem.landmarks
            if is_in_fov(end_step, l, problem.turning_radius*1.1, float(2pi), 0.0) # if in the orbit around a landmark
                landmark_in_fov = true
                push!(cost_vector, cost_vector[end])
                return cost_vector
            elseif is_in_fov(end_step, l, problem.vo_params.fovRadius, problem.vo_params.fovAngle, problem.vo_params.maxError)
                landmark_in_fov = true
                push!(cost_vector, 0.0)
                break
            end
        end

        if !landmark_in_fov
            cost,_ = VO.odometry_error(start_step, end_step, problem.mapped_features[1], problem.vo_params, max_cost)
            push!(cost_vector, cost_vector[end] + cost)
        end
        start_step = end_step
        s += step_size
    end
    return cost_vector
end

function sample_committed_trajectory(traj::CompositeTrajectory, t::F, vel::F) where {F}
 
    if t < traj.switch_time
        return sample_combined_dubins_path(traj.nominal_trajectory, t*vel)
    elseif t < traj.switch_time + traj.backup_time
        return sample_combined_dubins_path(traj.backup_trajectory, (t-traj.switch_time)*vel)
    else
        e, endpt = dubins_path_endpoint(traj.backup_trajectory[end])
        # force one orbit
        circle_path = DubinsPath(endpt,
                                 SVector(0.0,0.0,2pi*0.99), # make a path that is 99% of the circle
                                 traj.backup_trajectory[end].ρ,
                                 Dubins.LSL)
        e, x = dubins_path_sample(circle_path, (t- traj.switch_time - traj.backup_time)*vel)
        return x
    end
end