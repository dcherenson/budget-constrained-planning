module World
using StaticArrays
using LinearAlgebra
using Plots
abstract type AbstractUnsafeZone end

struct Circle{F} <: AbstractUnsafeZone
    center::SVector{2,F}
    radius::F
end

struct Rectangle{F} <: AbstractUnsafeZone
    center::SVector{2,F}
    width::F
    height::F
    angle::F
end

struct Cone{F} <: AbstractUnsafeZone
    center::SVector{2,F}
    radius::F
    orientation::F
    angular_width::F
end

function is_unsafe(zones::Vector{W}, x::SVector{3,F}) where {F,W<: AbstractUnsafeZone}
    for i in 1:length(zones)
        if is_unsafe(zones[i],x)
            return true
        end
    end
    return false
end

function is_unsafe(zone::Cone{F},  x::SVector{3,F}) where {F}
    # rotate the point into the cone frame
    x_rot = SVector(cos(zone.orientation)*x[1] - sin(zone.orientation)*x[2], sin(zone.orientation)*x[1] + cos(zone.orientation)*x[2])
    return norm(zone.center - x[SOneTo(2)]) < zone.radius && abs(angle_diff(atan(x_rot[2], x_rot[1]), zone.orientation)) < zone.angular_width/2
end

function is_unsafe(zone::Circle{F},  x::SVector{3,F}) where {F}
    # return norm(zone.center - x[SOneTo(2)]) < zone.radius
    return (zone.center[1] - x[1])^2 + (zone.center[2] - x[2])^2 < zone.radius^2
end

function is_unsafe(zone::Rectangle{F},  x::SVector{3,F}) where {F}
    # rotate the point into the rectangle frame
    diff = x[SOneTo(2)] - zone.center
    x_rot = SVector(cos(zone.angle)*diff[1] + sin(zone.angle)*diff[2], -sin(zone.angle)*diff[1] + cos(zone.angle)*diff[2])
    return abs(x_rot[1]) < zone.width/2 && abs(x_rot[2]) < zone.height/2    
end


function Plots.plot!(zones::Vector{W}; kwargs...) where {W <: AbstractUnsafeZone}
    for zone in zones
        plot!(zone; kwargs...)
    end
end

function Plots.plot!(zone::Circle{F}; kwargs...) where {F}
    θ = range(0, stop=2π, length=100)
    x = zone.center[1] .+ zone.radius*cos.(θ)
    y = zone.center[2] .+ zone.radius*sin.(θ)
    plot!(x, y; kwargs...)
    
    # Add 8 parallel lines at 45 degrees
    angle = π/4  # 45 degrees
    num_lines = 8
    
    # Direction perpendicular to the lines (for spacing)
    perp_x = -sin(angle)
    perp_y = cos(angle)
    
    # Direction along the lines
    dir_x = cos(angle)
    dir_y = sin(angle)
    
    # Spacing between lines
    spacing = 2 * zone.radius / (num_lines + 1)
    
    for i in 1:num_lines
        # Offset from center perpendicular to line direction
        offset = -zone.radius + i * spacing
        
        # Start point on the perpendicular
        cx = zone.center[1] + offset * perp_x
        cy = zone.center[2] + offset * perp_y
        
        # Calculate intersection length with circle at this offset
        # Using chord length formula: 2*sqrt(r^2 - d^2) where d is distance from center
        half_length = sqrt(max(0, zone.radius^2 - offset^2))
        
        # Line endpoints
        x1 = cx - half_length * dir_x
        y1 = cy - half_length * dir_y
        x2 = cx + half_length * dir_x
        y2 = cy + half_length * dir_y
        
        plot!([x1, x2], [y1, y2]; kwargs...)
    end
end

function Plots.plot!(zone::Rectangle{F}; kwargs...) where {F}
    x = [zone.center[1] - zone.width/2, zone.center[1] + zone.width/2, zone.center[1] + zone.width/2, zone.center[1] - zone.width/2, zone.center[1] - zone.width/2]
    y = [zone.center[2] - zone.height/2, zone.center[2] - zone.height/2, zone.center[2] + zone.height/2, zone.center[2] + zone.height/2, zone.center[2] - zone.height/2]
    R = [cos(zone.angle) -sin(zone.angle); sin(zone.angle) cos(zone.angle)]
    for i in 1:5
        p = R*[x[i] - zone.center[1], y[i] - zone.center[2]]
        x[i] = p[1] + zone.center[1]
        y[i] = p[2] + zone.center[2]
    end

    plot!(x, y; kwargs...)
end

function Plots.plot!(zone::Cone{F}; kwargs...) where {F}
    θ = range(zone.orientation - zone.angular_width/2, stop=zone.orientation + zone.angular_width/2, length=100)
    x = zone.center[1] .+ zone.radius*cos.(θ)
    y = zone.center[2] .+ zone.radius*sin.(θ)
    plot!(x, y; kwargs...)
end

end