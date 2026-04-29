# -----
# Helper Fucntions 
# -----
function active_status(s::PokemonState, actor::Symbol)
    if actor == :my
        return s.my_statuses[s.my_active]
    elseif actor == :opp
        return s.opp_statuses[s.opp_active]
    else
        error("Unknown actor: $actor")
    end
end

function active_speed(m::PokemonBattleMDP, s::PokemonState, actor::Symbol)
    pokemon = if actor == :my
        m.my_team[s.my_active]
    elseif actor == :opp
        m.opp_team[s.opp_active]
    else
        error("Unknown actor: $actor")
    end

    status = active_status(s, actor)

    if status == :paralyzed
        return max(pokemon.speed ÷ 2, 1)
    else
        return pokemon.speed
    end
end

function my_goes_first(m::PokemonBattleMDP, s::PokemonState)
    my_speed = active_speed(m, s, :my)
    opp_speed = active_speed(m, s, :opp)

    if my_speed > opp_speed
        return true
    elseif my_speed < opp_speed
        return false
    else
        return true  # tie-breaker: player goes first
    end
end

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
    if my_goes_first(m, s)
        return resolve_ordered_turn(m, s, my_action, :my, opp_action, :opp)
    else
        return resolve_ordered_turn(m, s, opp_action, :opp, my_action, :my)
    end
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

function resolve_ordered_turn(
    m::PokemonBattleMDP,
    s::PokemonState,
    first_action::Symbol,
    first_actor::Symbol,
    second_action::Symbol,
    second_actor::Symbol
)
    outcomes = Tuple{Float64, PokemonState}[]

    first_outcomes = apply_action(m, s, first_action, first_actor)

    for (p_first, s_after_first_raw) in first_outcomes
        s_after_first = auto_switch_fainted(m, s_after_first_raw)

        # If a whole team fainted, the turn ends.
        if all_fainted(s_after_first.my_hps) || all_fainted(s_after_first.opp_hps)
            push!(outcomes, (p_first, s_after_first))
            continue
        end

        # If the second actor's active Pokémon fainted, it cannot act.
        second_active_fainted = if second_actor == :my
            s_after_first_raw.my_hps[s_after_first_raw.my_active] <= 0
        else
            s_after_first_raw.opp_hps[s_after_first_raw.opp_active] <= 0
        end

        if second_active_fainted
            push!(outcomes, (p_first, s_after_first))
            continue
        end

        second_outcomes = apply_action(m, s_after_first, second_action, second_actor)

        for (p_second, sp_raw) in second_outcomes
            sp = auto_switch_fainted(m, sp_raw)
            push!(outcomes, (p_first * p_second, sp))
        end
    end

    return outcomes
end