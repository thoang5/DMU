function make_solver(; n_iterations::Int=100, depth::Int=100, exploration_constant::Float64=1.0)
    return MCTSSolver(
        n_iterations = n_iterations,
        depth = depth,
        exploration_constant = exploration_constant
    )
end

function make_planner(m::PokemonBattleMDP;
                      n_iterations::Int=100,
                      depth::Int=100,
                      exploration_constant::Float64=1.0)
    solver = make_solver(
        n_iterations=n_iterations,
        depth=depth,
        exploration_constant=exploration_constant
    )
    return solve(solver, m)
end