function resolve_turn(m::PokemonBattleMDP, s::PokemonState, my_action::PokemonAction, opp_action::PokemonAction)
    outcomes = Tuple{Float64, PokemonState}[]

    my_outcomes = apply_my_action(m, s, my_action)

    for (p_my, s_after_me) in my_outcomes
        if s_after_me.opp_hp <= 0
            push!(outcomes, (p_my, s_after_me))
            continue
        end

        opp_outcomes = apply_opp_action(m, s_after_me, opp_action)

        for (p_opp_turn, sp) in opp_outcomes
            push!(outcomes, (p_my * p_opp_turn, sp))
        end
    end

    return outcomes
end

function sample_opponent_action(s::PokemonState, rng::AbstractRNG)
    dist = opponent_policy(s)
    x = rand(rng)
    cumulative = 0.0

    for (a, p) in dist
        cumulative += p
        if x <= cumulative
            return a
        end
    end

    # fallback because of floating point roundoff
    return first(keys(dist))
end

function sample_turn_outcome(outcomes::Vector{Tuple{Float64, PokemonState}}, rng::AbstractRNG)
    if isempty(outcomes)
        error("sample_turn_outcome received an empty outcome list.")
    end

    x = rand(rng)
    cumulative = 0.0

    for (p, sp) in outcomes
        cumulative += p
        if x <= cumulative
            return sp
        end
    end

    return outcomes[end][2]
end


