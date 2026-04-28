function apply_action(m::PokemonBattleMDP, s::PokemonState, a::Symbol, actor::Symbol)

    if actor == :my
        pokemon = m.my_pokemon
        defender_type = m.opp_pokemon.type
        boost = s.my_boost
    elseif actor == :opp
        pokemon = m.opp_pokemon
        defender_type = m.my_pokemon.type
        boost = s.opp_boost
    else
        error("Unknown actor: $actor")
    end

    move = pokemon.moves[a]

    # Miss outcome
    miss_prob = 1.0 - move.accuracy

    # -------------------------
    # Damage move
    # -------------------------
    if move.category == :damage
        dmg = damage_amount(move, pokemon.type, defender_type, boost)

        if actor == :my
            hit_state = PokemonState(
                s.my_hp,
                max(s.opp_hp - dmg, 0),
                s.my_status,
                move.effect == :paralyze ? :paralyzed : s.opp_status,
                s.my_boost,
                s.opp_boost
            )
        else
            hit_state = PokemonState(
                max(s.my_hp - dmg, 0),
                s.opp_hp,
                move.effect == :paralyze ? :paralyzed : s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            )
        end

    # -------------------------
    # Healing move
    # -------------------------
    elseif move.category == :heal
        if actor == :my
            hit_state = PokemonState(
                min(s.my_hp + move.power, m.my_pokemon.max_hp),
                s.opp_hp,
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            )
        else
            hit_state = PokemonState(
                s.my_hp,
                min(s.opp_hp + move.power, m.opp_pokemon.max_hp),
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            )
        end

    # -------------------------
    # Boost move
    # -------------------------
    elseif move.category == :boost
        if actor == :my
            hit_state = PokemonState(
                s.my_hp,
                s.opp_hp,
                s.my_status,
                s.opp_status,
                min(s.my_boost + 1, 2),
                s.opp_boost
            )
        else
            hit_state = PokemonState(
                s.my_hp,
                s.opp_hp,
                s.my_status,
                s.opp_status,
                s.my_boost,
                min(s.opp_boost + 1, 2)
            )
        end

    # -------------------------
    # Status move
    # -------------------------
    elseif move.category == :status
        if move.effect == :paralyze
            if actor == :my
                hit_state = PokemonState(
                    s.my_hp,
                    s.opp_hp,
                    s.my_status,
                    :paralyzed,
                    s.my_boost,
                    s.opp_boost
                )
            else
                hit_state = PokemonState(
                    s.my_hp,
                    s.opp_hp,
                    :paralyzed,
                    s.opp_status,
                    s.my_boost,
                    s.opp_boost
                )
            end
        else
            hit_state = s
        end

    else
        hit_state = s
    end

    if move.accuracy >= 1.0
        return [(1.0, hit_state)]
    else
        return [
            (move.accuracy, hit_state),
            (miss_prob, s)
        ]
    end
end

function type_multiplier(attacking::Symbol, defending::Symbol)
    if attacking == :ElectricType && defending == :WaterType
        return 2.0
    elseif attacking == :ElectricType && defending == :GrassType
        return 0.5
    elseif attacking == :GrassType && defending == :WaterType
        return 2.0
    elseif attacking == :WaterType && defending == :GrassType
        return 0.5
    else
        return 1.0
    end
end

function damage_amount(move::Move, attacker_type::Symbol, defender_type::Symbol, boost::Int)
    mult = type_multiplier(move.move_type, defender_type)
    boost_bonus = boost * 10
    return max(round(Int, move.power * mult + boost_bonus), 0)
end

function apply_my_action(m::PokemonBattleMDP, s::PokemonState, a::Symbol)
    return apply_action(m, s, a, :my)
end

function apply_opp_action(m::PokemonBattleMDP, s::PokemonState, a::Symbol)
    return apply_action(m, s, a, :opp)
end