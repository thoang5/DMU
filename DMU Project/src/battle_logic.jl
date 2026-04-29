function first_alive_index(hps::Vector{Int})
    for i in eachindex(hps)
        if hps[i] > 0
            return i
        end
    end

    return nothing
end

function auto_switch_fainted(m::PokemonBattleMDP, s::PokemonState)
    new_my_active = s.my_active
    new_opp_active = s.opp_active

    # If my active Pokemon fainted but I still have another alive Pokemon,
    # switch to the first available alive Pokemon.
    if s.my_hps[s.my_active] <= 0 && !all_fainted(s.my_hps)
        new_my_active = first_alive_index(s.my_hps)
    end

    # If opponent active Pokemon fainted but opponent still has another alive Pokemon,
    # switch to the first available alive Pokemon.
    if s.opp_hps[s.opp_active] <= 0 && !all_fainted(s.opp_hps)
        new_opp_active = first_alive_index(s.opp_hps)
    end

    return PokemonState(
        copy(s.my_hps),
        copy(s.opp_hps),
        copy(s.my_statuses),
        copy(s.opp_statuses),
        new_my_active,
        new_opp_active,
        s.my_boost,
        s.opp_boost
    )
end

function resolve_turn(
    m::PokemonBattleMDP,
    s::PokemonState,
    my_action::Symbol,
    opp_action::Symbol
)
    outcomes = Tuple{Float64, PokemonState}[]

    my_outcomes = apply_my_action(m, s, my_action)

    for (p_my, s_after_me_raw) in my_outcomes

        # Auto-switch if my action caused the opponent active Pokemon to faint
        s_after_me = auto_switch_fainted(m, s_after_me_raw)

        # If opponent's entire team fainted, the turn ends immediately
        if all_fainted(s_after_me.opp_hps)
            push!(outcomes, (p_my, s_after_me))
            continue
        end

        # If opponent's active Pokemon fainted but they still have Pokemon left,
        # auto_switch_fainted already changed s_after_me.opp_active.
        # The opponent does NOT attack on this same turn after fainting.
        if s_after_me_raw.opp_hps[s_after_me_raw.opp_active] <= 0
            push!(outcomes, (p_my, s_after_me))
            continue
        end

        opp_outcomes = apply_opp_action(m, s_after_me, opp_action)

        for (p_opp_turn, sp_raw) in opp_outcomes

            # Auto-switch if opponent's action caused my active Pokemon to faint
            sp = auto_switch_fainted(m, sp_raw)

            push!(outcomes, (p_my * p_opp_turn, sp))
        end
    end

    return outcomes
end

function sample_opponent_action(m::PokemonBattleMDP, s::PokemonState, rng::AbstractRNG)
    dist = opponent_policy(m, s)

    x = rand(rng)
    cumulative = 0.0

    for (a, p) in dist
        cumulative += p
        if x <= cumulative
            return a
        end
    end

    return dist[end][1]
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