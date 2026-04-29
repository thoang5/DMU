struct Move
    name::Symbol
    move_type::Symbol
    category::Symbol   # :damage, :heal, :boost, :status
    power::Int
    accuracy::Float64
    effect::Symbol     # :none, :paralyze, :attack_boost, etc.
end

struct Pokemon
    name::Symbol
    type::Symbol
    max_hp::Int
    moves::Dict{Symbol, Move}
end

struct PokemonState
    my_hps::Vector{Int}
    opp_hps::Vector{Int}

    my_statuses::Vector{Symbol}
    opp_statuses::Vector{Symbol}

    my_active::Int
    opp_active::Int

    my_boost::Int
    opp_boost::Int
end

struct PokemonBattleMDP <: MDP{PokemonState, Symbol}
    my_team::Vector{Pokemon}
    opp_team::Vector{Pokemon}
    discount_factor::Float64
end