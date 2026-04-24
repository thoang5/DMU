function POMDPs.states(m::PokemonBattleMDP)
    states = PokemonState[]

    for my_hp in 0:m.my_max_hp
        for opp_hp in 0:m.opp_max_hp
            for my_status in 0:1
                for opp_status in 0:1
                    for my_boost in -2:2
                        for opp_boost in -2:2
                            push!(states, PokemonState(
                                my_hp,
                                opp_hp,
                                my_status,
                                opp_status,
                                my_boost,
                                opp_boost
                            ))
                        end
                    end
                end
            end
        end
    end

    return states
end

POMDPs.actions(m::PokemonBattleMDP) = [
    tackle,
    thunder_wave,
    swords_dance,
    recover
]

function opponent_policy(s::PokemonState)
    return [
        (tackle, 0.7),
        (recover, 0.2),
        (thunder_wave, 0.1)
    ]
end

POMDPs.discount(m::PokemonBattleMDP) = m.discount_factor

function POMDPs.reward(m::PokemonBattleMDP, s::PokemonState, a::PokemonAction, sp::PokemonState)
    if sp.opp_hp <= 0
        return 100.0
    elseif sp.my_hp <= 0
        return -100.0
    else
        damage_dealt = s.opp_hp - sp.opp_hp
        damage_taken = s.my_hp - sp.my_hp
        return damage_dealt - damage_taken
    end
end


function POMDPs.transition(m::PokemonBattleMDP, s::PokemonState, a::PokemonAction)
    next_states = PokemonState[]
    probs = Float64[]

    opp_dist = opponent_policy(s)

    for (opp_action, p_opp) in opp_dist
        outcomes = resolve_turn(m, s, a, opp_action)

        for (p_turn, sp) in outcomes
            push!(next_states, sp)
            push!(probs, p_opp * p_turn)
        end
    end

    return SparseCat(next_states, probs)
end

POMDPs.isterminal(m::PokemonBattleMDP, s::PokemonState) = s.my_hp <= 0 || s.opp_hp <= 0

function POMDPs.gen(m::PokemonBattleMDP, s::PokemonState, a::PokemonAction, rng::AbstractRNG)
    opp_action = sample_opponent_action(s, rng)

    outcomes = resolve_turn(m, s, a, opp_action)

    sp = sample_turn_outcome(outcomes, rng)

    r = reward(m, s, a, sp)

    return (sp=sp, r=r)
end