using Statistics

function initial_pokemon_state(m::PokemonBattleMDP)
    return PokemonState(
        m.my_pokemon.max_hp,
        m.opp_pokemon.max_hp,
        :none,
        :none,
        0,
        0
    )
end

function run_mcts_episode(
    m::PokemonBattleMDP;
    n_iterations::Int = 100,
    depth::Int = 10,
    exploration_constant::Float64 = 1.0,
    max_turns::Int = 50,
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

    while !isterminal(m, s) && turn <= max_turns
        a = action(planner, s)
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

    outcome = if s.opp_hp <= 0 && s.my_hp <= 0
        :tie
    elseif s.opp_hp <= 0
        :win
    elseif s.my_hp <= 0
        :loss
    else
        :max_turns
    end

    return total_reward, outcome, turn - 1
end

function evaluate_mcts(
    m::PokemonBattleMDP;
    n_episodes::Int = 100,
    n_iterations::Int = 100,
    depth::Int = 10,
    exploration_constant::Float64 = 1.0,
    max_turns::Int = 50
)
    returns = Float64[]
    outcomes = Symbol[]
    turns = Int[]

    for ep in 1:n_episodes
        total_reward, outcome, n_turns = run_mcts_episode(
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
    end

    avg_return = mean(returns)
    win_rate = count(outcomes .== :win) / n_episodes
    avg_turns = mean(turns)

    return returns, outcomes, turns, avg_return, win_rate, avg_turns
end

function evaluate_mcts_sweep(
    m::PokemonBattleMDP;
    iteration_list = [1, 5, 10, 25, 50, 100, 250, 500],
    n_episodes::Int = 100,
    depth::Int = 10,
    exploration_constant::Float64 = 1.0,
    max_turns::Int = 50
)
    avg_returns = Float64[]
    return_sems = Float64[]
    win_rates = Float64[]
    avg_turns_list = Float64[]

    for n_iter in iteration_list
        println("Evaluating MCTS with n_iterations = ", n_iter)

        returns, outcomes, turns, avg_return, win_rate, avg_turns = evaluate_mcts(
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
    end

    return iteration_list, avg_returns, return_sems, win_rates, avg_turns_list
end

function first_action_frequency_sweep(
    m::PokemonBattleMDP;
    target_action::Symbol = :move2,
    iteration_list = [1, 5, 10, 25, 50, 100, 250, 500],
    n_trials::Int = 100,
    depth::Int = 10,
    exploration_constant::Float64 = 1.0
)
    frequencies = Float64[]

    s0 = initial_pokemon_state(m)

    for n_iter in iteration_list
        println("Testing first action choice with n_iterations = ", n_iter)

        solver = MCTSSolver(
            n_iterations = n_iter,
            depth = depth,
            exploration_constant = exploration_constant
        )

        planner = solve(solver, m)

        chosen_actions = Symbol[]

        for trial in 1:n_trials
            a = action(planner, s0)
            push!(chosen_actions, a)
        end

        freq = count(chosen_actions .== target_action) / n_trials
        push!(frequencies, freq)
    end

    return iteration_list, frequencies
end