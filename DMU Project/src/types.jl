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
    my_hp::Int
    opp_hp::Int
    my_status::Symbol
    opp_status::Symbol
    my_boost::Int
    opp_boost::Int
end

struct PokemonBattleMDP <: MDP{PokemonState, Symbol}
    my_pokemon::Pokemon
    opp_pokemon::Pokemon
    discount_factor::Float64
end