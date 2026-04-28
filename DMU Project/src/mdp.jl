function POMDPs.initialstate(m::PokemonBattleMDP)
    return Deterministic(PokemonState(
        m.my_pokemon.max_hp,
        m.opp_pokemon.max_hp,
        :none,
        :none,
        0,
        0
    ))
end

POMDPs.actions(m::PokemonBattleMDP) = sort(collect(keys(m.my_pokemon.moves)))

function opponent_policy(m::PokemonBattleMDP, s::PokemonState)
    return [
        (:move1, 0.1),
        (:move2, 0.1),
        (:move3, 0.8)
    ]
end

POMDPs.discount(m::PokemonBattleMDP) = m.discount_factor

POMDPs.isterminal(m::PokemonBattleMDP, s::PokemonState) =
    s.my_hp <= 0 || s.opp_hp <= 0

function POMDPs.reward(m::PokemonBattleMDP, s::PokemonState, a::Symbol, sp::PokemonState)
    damage_dealt = s.opp_hp - sp.opp_hp
    damage_taken = s.my_hp - sp.my_hp

    if sp.opp_hp <= 0 && sp.my_hp <= 0
        return 0.0
    elseif sp.opp_hp <= 0
        return 1000.0
    elseif sp.my_hp <= 0
        return -1000.0
    else
        return -1.0 
    end
end

function POMDPs.transition(m::PokemonBattleMDP, s::PokemonState, a::Symbol)
    next_states = PokemonState[]
    probs = Float64[]

    opp_dist = opponent_policy(m, s)

    for (opp_action, p_opp) in opp_dist
        outcomes = resolve_turn(m, s, a, opp_action)

        for (p_turn, sp) in outcomes
            push!(next_states, sp)
            push!(probs, p_opp * p_turn)
        end
    end

    return SparseCat(next_states, probs)
end

function POMDPs.gen(m::PokemonBattleMDP, s::PokemonState, a::Symbol, rng::AbstractRNG)
    opp_action = sample_opponent_action(m, s, rng)

    outcomes = resolve_turn(m, s, a, opp_action)

    sp = sample_turn_outcome(outcomes, rng)

    r = reward(m, s, a, sp)

    return (sp=sp, r=r)
end