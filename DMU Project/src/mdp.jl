# -------------------------
# Helper functions
# -------------------------

function all_fainted(hps::Vector{Int})
    return all(hp -> hp <= 0, hps)
end

function active_my_pokemon(m::PokemonBattleMDP, s::PokemonState)
    return m.my_team[s.my_active]
end

function active_opp_pokemon(m::PokemonBattleMDP, s::PokemonState)
    return m.opp_team[s.opp_active]
end

function active_my_hp(s::PokemonState)
    return s.my_hps[s.my_active]
end

function active_opp_hp(s::PokemonState)
    return s.opp_hps[s.opp_active]
end

function best_type_multiplier_against(attacker::Pokemon, defender::Pokemon)
    best_mult = 0.0

    for move in values(attacker.moves)
        if move.category == :damage
            mult = type_multiplier(move.move_type, defender.type)
            best_mult = max(best_mult, mult)
        end
    end

    return best_mult
end

# -------------------------
# Initial state
# -------------------------

function POMDPs.initialstate(m::PokemonBattleMDP)
    return Deterministic(PokemonState(
        [p.max_hp for p in m.my_team],
        [p.max_hp for p in m.opp_team],

        fill(:none, length(m.my_team)),
        fill(:none, length(m.opp_team)),

        1,
        1,

        0,
        0
    ))
end

# -------------------------
# Actions
# -------------------------

function POMDPs.actions(m::PokemonBattleMDP, s::PokemonState)
    active_pokemon = m.my_team[s.my_active]

    # Move actions for the active Pokemon
    move_actions = sort(collect(keys(active_pokemon.moves)))

    # Switch actions to any alive, non-active Pokemon
    switch_actions = Symbol[]

    for i in eachindex(m.my_team)
        if i != s.my_active && s.my_hps[i] > 0
            push!(switch_actions, Symbol("switch", i))
        end
    end

    return vcat(move_actions, switch_actions)
end

# -------------------------
# Opponent policy
# -------------------------

function opponent_policy(m::PokemonBattleMDP, s::PokemonState)
    opp_pokemon = m.opp_team[s.opp_active]
    my_pokemon = m.my_team[s.my_active]

    actions = Symbol[]
    weights = Float64[]

    # -------------------------
    # Move action weights
    # -------------------------
    for a in sort(collect(keys(opp_pokemon.moves)))
        move = opp_pokemon.moves[a]

        weight = 1.0

        if move.category == :damage
            mult = type_multiplier(move.move_type, my_pokemon.type)

            # Prefer super-effective moves
            if mult > 1.0
                weight += 5.0
            elseif mult < 1.0
                weight -= 0.5
            end

            # Prefer stronger moves
            weight += move.power / 100.0

            # Prefer accurate moves
            weight *= move.accuracy

        elseif move.category == :heal
            # Heal more often when low HP
            hp_ratio = s.opp_hps[s.opp_active] / opp_pokemon.max_hp

            if hp_ratio < 0.4
                weight += 4.0
            else
                weight += 0.2
            end

        elseif move.category == :status
            # Use status if target does not already have a status
            if s.my_statuses[s.my_active] == :none
                weight += 2.0
            else
                weight += 0.1
            end

        elseif move.category == :boost
            # Boost if not already boosted
            if s.opp_boost < 2
                weight += 1.5
            else
                weight += 0.1
            end
        end

        push!(actions, a)
        push!(weights, max(weight, 0.1))
    end

    # -------------------------
    # Switch action weights
    # -------------------------
    for i in eachindex(m.opp_team)
        if i != s.opp_active && s.opp_hps[i] > 0
            switch_action = Symbol("switch", i)
            switch_pokemon = m.opp_team[i]

            current_bad_matchup = best_type_multiplier_against(my_pokemon, opp_pokemon)
            new_bad_matchup = best_type_multiplier_against(my_pokemon, switch_pokemon)

            current_attack = best_type_multiplier_against(opp_pokemon, my_pokemon)
            new_attack = best_type_multiplier_against(switch_pokemon, my_pokemon)

            weight = 0.5

            # Switch if current active is threatened and new switch-in is safer
            if current_bad_matchup > new_bad_matchup
                weight += 3.0
            end

            # Switch if the new Pokemon can hit my active Pokemon harder
            if new_attack > current_attack
                weight += 2.0
            end

            push!(actions, switch_action)
            push!(weights, max(weight, 0.1))
        end
    end

    total_weight = sum(weights)

    return [(actions[i], weights[i] / total_weight) for i in eachindex(actions)]
end

# -------------------------
# Discount
# -------------------------

POMDPs.discount(m::PokemonBattleMDP) = m.discount_factor

# -------------------------
# Terminal condition
# -------------------------

POMDPs.isterminal(m::PokemonBattleMDP, s::PokemonState) =
    all_fainted(s.my_hps) || all_fainted(s.opp_hps)

# -------------------------
# Reward
# -------------------------

function POMDPs.reward(m::PokemonBattleMDP, s::PokemonState, a::Symbol, sp::PokemonState)
    if all_fainted(sp.opp_hps) && all_fainted(sp.my_hps)
        return 0.0
    elseif all_fainted(sp.opp_hps)
        return 1000.0
    elseif all_fainted(sp.my_hps)
        return -1000.0
    else
        return -1.0
    end
end

# -------------------------
# Transition model
# -------------------------

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

# -------------------------
# Generative model
# -------------------------

function POMDPs.gen(m::PokemonBattleMDP, s::PokemonState, a::Symbol, rng::AbstractRNG)
    opp_action = sample_opponent_action(m, s, rng)

    outcomes = resolve_turn(m, s, a, opp_action)

    sp = sample_turn_outcome(outcomes, rng)

    r = reward(m, s, a, sp)

    return (sp=sp, r=r)
end