using .World
using StaticArrays
using Dubins

@kwdef struct NominalPlannerProblem{F, VW}
    domain::Tuple{SVector{3,F}, SVector{3,F}} # rectangle defined by opposite corners
    turning_radius::F = 10.0
    max_velocity::F = 10.0
    unsafe_zones::VW
    goal::SVector{3,F}
end

struct NominalPlanner{TP}
    prob::TP # NominalPlannerProblem{Float64}
end

function initialize_nominal_planner(goal::TV, domain::Tuple{TV, TV}, zones::Vector{W}, max_iter=nothing) where {TV, W <: World.AbstractUnsafeZone}
    # max_iter is ignored but kept for API compatibility
    P = NominalPlannerProblem(domain = domain, unsafe_zones = zones, goal = goal)
    return NominalPlanner(P)
end

function query_nominal_planner(planner::NominalPlanner, start)
    # Simply compute the Dubins shortest path from start to goal
    errcode, path = Dubins.dubins_shortest_path(start, planner.prob.goal, planner.prob.turning_radius, 1e-3)
    
    if errcode != Dubins.EDUBOK
        return false, SVector{3,Float64}[]
    end
    
    # Return the start and goal as the path waypoints
    return true, [start, planner.prob.goal]
end