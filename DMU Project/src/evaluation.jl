function evaluate_mcts(m::PokemonBattleMDP; n_episodes::Int = 100)
    returns = Float64[]

    for ep in 1:n_episodes
        _, total_reward = play_mcts_game(m; seed=ep, verbose=false)
        push!(returns, total_reward)
    end

    avg_return = sum(returns) / length(returns)

    return returns, avg_return
end