function play_mcts_game(m::PokemonBattleMDP;
                        start_state::PokemonState = PokemonState(10, 10, 0, 0, 0, 0),
                        max_turns::Int = 50,
                        seed::Int = 1,
                        verbose::Bool = true)

    solver = MCTSSolver(
        n_iterations = 100,
        depth = 10,
        exploration_constant = 1.0
    )

    planner = solve(solver, m)
    rng = MersenneTwister(seed)

    s = start_state
    total_reward = 0.0
    turn = 1

    if verbose
        println("Starting game")
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

        if s.opp_hp <= 0 && s.my_hp <= 0
            println("Result: tie")
        elseif s.opp_hp <= 0
            println("Result: you win")
        elseif s.my_hp <= 0
            println("Result: you lose")
        else
            println("Result: max turns reached")
        end
    end

    return s, total_reward
end