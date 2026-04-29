function initial_pokemon_state(m::PokemonBattleMDP)
    return PokemonState(
        [p.max_hp for p in m.my_team],
        [p.max_hp for p in m.opp_team],

        fill(:none, length(m.my_team)),
        fill(:none, length(m.opp_team)),

        1,  # my active Pokemon
        1,  # opponent active Pokemon

        0,
        0
    )
end

function play_mcts_game(m::PokemonBattleMDP;
                        start_state::PokemonState = initial_pokemon_state(m),
                        max_turns::Int = 50,
                        seed::Int = 1,
                        verbose::Bool = true,
                        n_iterations::Int = 100,
                        depth::Int = 10,
                        exploration_constant::Float64 = 1.0)

    solver = MCTSSolver(
        n_iterations = n_iterations,
        depth = depth,
        exploration_constant = exploration_constant
    )

    planner = solve(solver, m)
    rng = MersenneTwister(seed)

    s = start_state
    total_reward = 0.0
    turn = 1

    if verbose
        println("Starting game")
        println("MCTS settings:")
        println("  n_iterations = ", n_iterations)
        println("  depth = ", depth)
        println("  exploration_constant = ", exploration_constant)
        println("Initial state: ", s)
        println()
    end

    while !isterminal(m, s) && turn <= max_turns
        a = action(planner, s)
        result = gen(m, s, a, rng)

        sp = result.sp
        r = result.r

        if verbose
            println("Turn ", turn)
            println("Current state: ", s)
            println("MCTS chose: ", a)
            println("Next state: ", sp)
            println("Reward: ", r)
            println()
        end

        s = sp
        total_reward += r
        turn += 1
    end

    if verbose
        println("Game ended.")
        println("Final state: ", s)
        println("Total reward: ", total_reward)

        if all_fainted(s.opp_hps) && all_fainted(s.my_hps)
            println("Result: tie")
        elseif all_fainted(s.opp_hps)
            println("Result: you win")
        elseif all_fainted(s.my_hps)
            println("Result: you lose")
        else
            println("Result: max turns reached")
        end
    end

    return s, total_reward
end