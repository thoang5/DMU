function is_switch_action(a::Symbol)
    return startswith(String(a), "switch")
end

function switch_index(a::Symbol)
    return parse(Int, replace(String(a), "switch" => ""))
end

# -------------------------
# Type effectiveness
# -------------------------

function type_multiplier(attacking::Symbol, defending::Symbol)
    if attacking == :ElectricType && defending == :WaterType
        return 2.0
    elseif attacking == :ElectricType && defending == :GrassType
        return 0.5
    elseif attacking == :ElectricType && defending == :GroundType
        return 0.0
    elseif attacking == :ElectricType && defending == :ElectricType
        return 0.5

    elseif attacking == :GrassType && defending == :WaterType
        return 2.0
    elseif attacking == :GrassType && defending == :FireType
        return 0.5
    elseif attacking == :GrassType && defending == :GrassType
        return 0.5

    elseif attacking == :WaterType && defending == :FireType
        return 2.0
    elseif attacking == :WaterType && defending == :GrassType
        return 0.5
    elseif attacking == :WaterType && defending == :WaterType
        return 0.5

    elseif attacking == :FireType && defending == :GrassType
        return 2.0
    elseif attacking == :FireType && defending == :WaterType
        return 0.5
    elseif attacking == :FireType && defending == :FireType
        return 0.5

    elseif attacking == :GroundType && defending == :ElectricType
        return 2.0
    elseif attacking == :GroundType && defending == :GrassType
        return 0.5

    else
        return 1.0
    end
end

function damage_amount(move::Move, attacker_type::Symbol, defender_type::Symbol, boost::Int)
    mult = type_multiplier(move.move_type, defender_type)

    # boost = -2, -1, 0, 1, 2
    # boost multiplier = 0.5, 0.75, 1.0, 1.25, 1.5
    boost_mult = 1.0 + boost

    return max(round(Int, move.power * mult * boost_mult), 0)
end

function apply_action(m::PokemonBattleMDP, s::PokemonState, a::Symbol, actor::Symbol)


    if is_switch_action(a)
        new_active = switch_index(a)

        if actor == :my
            if new_active == s.my_active || s.my_hps[new_active] <= 0
                return [(1.0, s)]  # invalid switch, no change
            end

            return [(1.0, PokemonState(
                copy(s.my_hps),
                copy(s.opp_hps),
                copy(s.my_statuses),
                copy(s.opp_statuses),
                new_active,
                s.opp_active,
                0,              # reset my boost on switch
                s.opp_boost
            ))]

        elseif actor == :opp
            if new_active == s.opp_active || s.opp_hps[new_active] <= 0
                return [(1.0, s)]
            end

            return [(1.0, PokemonState(
                copy(s.my_hps),
                copy(s.opp_hps),
                copy(s.my_statuses),
                copy(s.opp_statuses),
                s.my_active,
                new_active,
                s.my_boost,
                0               # reset opponent boost on switch
            ))]
        end
    end

    if actor == :my
        attacker = m.my_team[s.my_active]
        defender = m.opp_team[s.opp_active]
        boost = s.my_boost
    elseif actor == :opp
        attacker = m.opp_team[s.opp_active]
        defender = m.my_team[s.my_active]
        boost = s.opp_boost
    else
        error("Unknown actor: $actor")
    end

    move = attacker.moves[a]
    miss_prob = 1.0 - move.accuracy

    # Copy state pieces so we can modify them safely
    new_my_hps = copy(s.my_hps)
    new_opp_hps = copy(s.opp_hps)
    new_my_statuses = copy(s.my_statuses)
    new_opp_statuses = copy(s.opp_statuses)

    new_my_boost = s.my_boost
    new_opp_boost = s.opp_boost

    # -------------------------
    # Damage move
    # -------------------------
    if move.category == :damage
        dmg = damage_amount(move, attacker.type, defender.type, boost)

        if actor == :my
            new_opp_hps[s.opp_active] = max(s.opp_hps[s.opp_active] - dmg, 0)

            if move.effect == :paralyze
                new_opp_statuses[s.opp_active] = :paralyzed
            end

        elseif actor == :opp
            new_my_hps[s.my_active] = max(s.my_hps[s.my_active] - dmg, 0)

            if move.effect == :paralyze
                new_my_statuses[s.my_active] = :paralyzed
            end
        end

    # -------------------------
    # Healing move
    # -------------------------
    elseif move.category == :heal
        if actor == :my
            new_my_hps[s.my_active] = min(
                s.my_hps[s.my_active] + move.power,
                attacker.max_hp
            )
        elseif actor == :opp
            new_opp_hps[s.opp_active] = min(
                s.opp_hps[s.opp_active] + move.power,
                attacker.max_hp
            )
        end

    # -------------------------
    # Boost move
    # -------------------------
    elseif move.category == :boost
        if actor == :my
            new_my_boost = min(s.my_boost + 1, 2)
        elseif actor == :opp
            new_opp_boost = min(s.opp_boost + 1, 2)
        end

    # -------------------------
    # Status move
    # -------------------------
    elseif move.category == :status
        if move.effect == :paralyze
            if actor == :my
                new_opp_statuses[s.opp_active] = :paralyzed
            elseif actor == :opp
                new_my_statuses[s.my_active] = :paralyzed
            end
        end
    end

    hit_state = PokemonState(
        new_my_hps,
        new_opp_hps,
        new_my_statuses,
        new_opp_statuses,
        s.my_active,
        s.opp_active,
        new_my_boost,
        new_opp_boost
    )

    if move.accuracy >= 1.0
        return [(1.0, hit_state)]
    else
        return [
            (move.accuracy, hit_state),
            (miss_prob, s)
        ]
    end
end

function apply_my_action(m::PokemonBattleMDP, s::PokemonState, a::Symbol)
    return apply_action(m, s, a, :my)
end

function apply_opp_action(m::PokemonBattleMDP, s::PokemonState, a::Symbol)
    return apply_action(m, s, a, :opp)
end