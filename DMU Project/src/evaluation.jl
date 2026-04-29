using Statistics
using Random

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

function initial_pokemon_state(m::PokemonBattleMDP)
    return PokemonState(
        [p.max_hp for p in m.my_team],
        [p.max_hp for p in m.opp_team],
        fill(:none, length(m.my_team)),
        fill(:none, length(m.opp_team)),
        1,
        1,
        0,
        0
    )
end

function run_mcts_episode(
    m::PokemonBattleMDP;
    n_iterations::Int = 100,
    depth::Int = 10,
    exploration_constant::Float64 = 100.0,
    max_turns::Int = 100,
    seed::Int = 1,
    verbose::Bool = false
)
    solver = MCTSSolver(
        n_iterations = n_iterations,
        depth = depth,
        exploration_constant = exploration_constant
    )

    planner = solve(solver, m)
    rng = MersenneTwister(seed)

    s = initial_pokemon_state(m)
    total_reward = 0.0
    turn = 1
    n_switches = 0

    start_time = time()

    while !isterminal(m, s) && turn <= max_turns
        a = action(planner, s)

        if startswith(String(a), "switch")
            n_switches += 1
        end

        result = gen(m, s, a, rng)

        sp = result.sp
        r = result.r

        if verbose
            println("Turn ", turn)
            println("State: ", s)
            println("MCTS chose: ", a)
            println("Next state: ", sp)
            println("Reward: ", r)
            println()
        end

        s = sp
        total_reward += r
        turn += 1
    end

    elapsed_time = time() - start_time

    outcome = if all_fainted(s.opp_hps) && all_fainted(s.my_hps)
        :tie
    elseif all_fainted(s.opp_hps)
        :win
    elseif all_fainted(s.my_hps)
        :loss
    else
        :max_turns
    end

    return total_reward, outcome, turn - 1, n_switches, elapsed_time
end

function evaluate_mcts(
    m::PokemonBattleMDP;
    n_episodes::Int = 100,
    n_iterations::Int = 100,
    depth::Int = 10,
    exploration_constant::Float64 = 100.0,
    max_turns::Int = 100
)
    returns = Float64[]
    outcomes = Symbol[]
    turns = Int[]
    switch_counts = Int[]
    runtimes = Float64[]

    for ep in 1:n_episodes
        total_reward, outcome, n_turns, n_switches, elapsed_time = run_mcts_episode(
            m;
            n_iterations = n_iterations,
            depth = depth,
            exploration_constant = exploration_constant,
            max_turns = max_turns,
            seed = ep,
            verbose = false
        )

        push!(returns, total_reward)
        push!(outcomes, outcome)
        push!(turns, n_turns)
        push!(switch_counts, n_switches)
        push!(runtimes, elapsed_time)
    end

    avg_return = mean(returns)
    win_rate = count(outcomes .== :win) / n_episodes
    avg_turns = mean(turns)
    avg_switches = mean(switch_counts)
    avg_runtime = mean(runtimes)

    return returns, outcomes, turns, switch_counts, runtimes,
           avg_return, win_rate, avg_turns, avg_switches, avg_runtime
end

function evaluate_mcts_sweep(
    m::PokemonBattleMDP;
    iteration_list = [1, 5, 10, 25, 50, 100, 250, 500],
    n_episodes::Int = 100,
    depth::Int = 10,
    exploration_constant::Float64 = 100.0,
    max_turns::Int = 100
)
    avg_returns = Float64[]
    return_sems = Float64[]
    win_rates = Float64[]
    avg_turns_list = Float64[]
    avg_switches_list = Float64[]
    avg_runtime_list = Float64[]

    for n_iter in iteration_list
        println("Evaluating MCTS with n_iterations = ", n_iter)

        returns, outcomes, turns, switch_counts, runtimes,
        avg_return, win_rate, avg_turns, avg_switches, avg_runtime = evaluate_mcts(
            m;
            n_episodes = n_episodes,
            n_iterations = n_iter,
            depth = depth,
            exploration_constant = exploration_constant,
            max_turns = max_turns
        )

        sem = length(returns) > 1 ? std(returns) / sqrt(length(returns)) : 0.0

        push!(avg_returns, avg_return)
        push!(return_sems, sem)
        push!(win_rates, win_rate)
        push!(avg_turns_list, avg_turns)
        push!(avg_switches_list, avg_switches)
        push!(avg_runtime_list, avg_runtime)
    end

    return iteration_list,
           avg_returns,
           return_sems,
           win_rates,
           avg_turns_list,
           avg_switches_list,
           avg_runtime_list
end

function first_action_frequency_sweep(
    m::PokemonBattleMDP;
    target_action::Symbol,
    iteration_list = [1, 5, 10, 25, 50, 100, 250, 500],
    n_trials::Int = 100,
    depth::Int = 10,
    exploration_constant::Float64 = 100.0
)
    frequencies = Float64[]

    s0 = initial_pokemon_state(m)

    for n_iter in iteration_list
        println("Testing first action choice with n_iterations = ", n_iter)

        chosen_actions = Symbol[]

        for trial in 1:n_trials
            solver = MCTSSolver(
                n_iterations = n_iter,
                depth = depth,
                exploration_constant = exploration_constant
            )

            planner = solve(solver, m)
            a = action(planner, s0)

            push!(chosen_actions, a)
        end

        freq = count(chosen_actions .== target_action) / n_trials
        push!(frequencies, freq)
    end

    return iteration_list, frequencies
end

function greedy_heuristic_action(m::PokemonBattleMDP, s::PokemonState)
    legal_actions = POMDPs.actions(m, s)

    best_action = legal_actions[1]
    best_score = -Inf

    attacker = m.my_team[s.my_active]
    defender = m.opp_team[s.opp_active]

    for a in legal_actions
        score = 0.0

        if is_switch_action(a)
            new_active = switch_index(a)
            switch_pokemon = m.my_team[new_active]

            # Prefer switching if new Pokemon has a better type matchup
            current_best = best_type_multiplier_against(attacker, defender)
            switch_best = best_type_multiplier_against(switch_pokemon, defender)

            score = 20.0 * (switch_best - current_best)

        else
            move = attacker.moves[a]

            if move.category == :damage
                dmg = damage_amount(move, attacker.type, defender.type, s.my_boost)
                score = move.accuracy * dmg

            elseif move.category == :heal
                missing_hp = attacker.max_hp - s.my_hps[s.my_active]
                score = min(40, missing_hp)

            elseif move.category == :boost
                # Encourage Swords Dance if not already boosted
                score = s.my_boost < 6 ? 50.0 : 0.0

            elseif move.category == :status
                if move.effect == :paralyze && s.opp_statuses[s.opp_active] == :none
                    score = 45.0
                else
                    score = 0.0
                end
            end
        end

        if score > best_score
            best_score = score
            best_action = a
        end
    end

    return best_action
end

function run_greedy_episode(
    m::PokemonBattleMDP;
    max_turns::Int = 150,
    seed::Int = 1
)
    rng = MersenneTwister(seed)
    s = initial_pokemon_state(m)

    total_reward = 0.0
    turn = 1
    n_switches = 0

    while !isterminal(m, s) && turn <= max_turns
        a = greedy_heuristic_action(m, s)

        if startswith(String(a), "switch")
            n_switches += 1
        end

        result = gen(m, s, a, rng)

        s = result.sp
        total_reward += result.r
        turn += 1
    end

    outcome = if all_fainted(s.opp_hps) && all_fainted(s.my_hps)
        :tie
    elseif all_fainted(s.opp_hps)
        :win
    elseif all_fainted(s.my_hps)
        :loss
    else
        :max_turns
    end

    return total_reward, outcome, turn - 1, n_switches
end

function evaluate_greedy(
    m::PokemonBattleMDP;
    n_episodes::Int = 100,
    max_turns::Int = 150
)
    returns = Float64[]
    outcomes = Symbol[]
    turns = Int[]
    switches = Int[]

    for ep in 1:n_episodes
        total_reward, outcome, n_turns, n_switches = run_greedy_episode(
            m;
            max_turns = max_turns,
            seed = ep
        )

        push!(returns, total_reward)
        push!(outcomes, outcome)
        push!(turns, n_turns)
        push!(switches, n_switches)
    end

    avg_return = mean(returns)
    win_rate = count(outcomes .== :win) / n_episodes
    avg_turns = mean(turns)
    avg_switches = mean(switches)

    return avg_return, win_rate, avg_turns, avg_switches
end

