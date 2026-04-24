struct PokemonState
    my_hp::Int
    opp_hp::Int
    my_status::Int
    opp_status::Int
    my_boost::Int
    opp_boost::Int
end

@enum PokemonType begin
    NormalType
    ElectricType
    WaterType
    GrassType
end

@enum PokemonAction begin
    tackle
    thunder_wave
    swords_dance
    recover
end

struct PokemonBattleMDP <: MDP{PokemonState, PokemonAction}
    discount_factor::Float64
    my_type::PokemonType
    opp_type::PokemonType
    my_max_hp::Int
    opp_max_hp::Int
    my_attack::Int
    opp_attack::Int
end