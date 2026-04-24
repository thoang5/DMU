function apply_my_action(m::PokemonBattleMDP, s::PokemonState, a::PokemonAction)
    if a == tackle || a == thunderbolt
        dmg = damage_amount(a, m.my_type, m.opp_type)
        return [
            (0.9, PokemonState(
                s.my_hp,
                max(s.opp_hp - dmg, 0),
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            )),
            (0.1, s)
        ]

    elseif a == swords_dance
        return [
            (1.0, PokemonState(
                s.my_hp,
                s.opp_hp,
                s.my_status,
                s.opp_status,
                min(s.my_boost + 1, 2),
                s.opp_boost
            ))
        ]

    elseif a == recover
        return [
            (1.0, PokemonState(
                min(s.my_hp + 2, 10),
                s.opp_hp,
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            ))
        ]
    end
end

function apply_opp_action(m::PokemonBattleMDP, s::PokemonState, a::PokemonAction)
    if a == tackle || a == thunderbolt
        dmg = damage_amount(a, m.opp_type, m.my_type)
        return [
            (0.9, PokemonState(
                max(s.my_hp - dmg, 0),
                s.opp_hp,
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            )),
            (0.1, s)
        ]

    elseif a == swords_dance
        return [
            (1.0, PokemonState(
                s.my_hp,
                s.opp_hp,
                s.my_status,
                s.opp_status,
                s.my_boost,
                min(s.opp_boost + 1, 2)
            ))
        ]

    elseif a == recover
        return [
            (1.0, PokemonState(
                s.my_hp,
                min(s.opp_hp + 2, 10),
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            ))
        ]
    end
end

function move_type(a::PokemonAction)
    if a == tackle
        return NormalType
    elseif a == thunderbolt
        return ElectricType
    elseif a == swords_dance
        return NormalType
    elseif a == recover
        return NormalType
    end
end

function base_power(a::PokemonAction)
    if a == tackle
        return 2
    elseif a == thunderbolt
        return 2
    else
        return 0
    end
end

function type_multiplier(attacking::PokemonType, defending::PokemonType)
    if attacking == ElectricType && defending == WaterType
        return 2.0
    elseif attacking == ElectricType && defending == GrassType
        return 0.5
    elseif attacking == GrassType && defending == WaterType
        return 2.0
    elseif attacking == WaterType && defending == GrassType
        return 0.5
    else
        return 1.0
    end
end

function damage_amount(a::PokemonAction, attacker_type::PokemonType, defender_type::PokemonType)
    base = base_power(a)
    mult = type_multiplier(move_type(a), defender_type)
    return round(Int, base * mult)
end