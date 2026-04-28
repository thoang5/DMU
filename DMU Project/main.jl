include("src/PokemonBattle.jl")
using .PokemonBattle
using Plots
const PB = PokemonBattle

# Moves
tackle = PB.Move(:tackle, :NormalType, :damage, 40, 1.0, :none)
thunderbolt = PB.Move(:thunderbolt, :ElectricType, :damage, 100, 1.0, :none)
water_gun = PB.Move(:water_gun, :WaterType, :damage, 60, 1.0, :none)
grass_knot = PB.Move(:grass_knot, :GrassType, :damage, 120, 1.0, :none)
bite = PB.Move(:bite, :DarkType, :damage, 100, 1.0, :none)
nuzzle = PB.Move(:nuzzle, :ElectricType, :damage, 40, 1.0, :paralyze)

# Pokemon
pikachu = PB.Pokemon(
    :Pikachu,
    :ElectricType,
    1000,
    Dict(
        :move1 => tackle,
        :move2 => thunderbolt,
        :move3 => nuzzle,
        :move4 => grass_knot
    )
)

squirtle = PB.Pokemon(
    :Squirtle,
    :WaterType,
    1000,
    Dict(
        :move1 => tackle,
        :move2 => water_gun,
        :move3 => bite
    )
)

# MDP
m = PB.PokemonBattleMDP(
    pikachu,
    squirtle,
    0.95
)

s0 = PB.PokemonState(
    m.my_pokemon.max_hp,
    m.opp_pokemon.max_hp,
    :none,
    :none,
    0,
    0
)

# -------------------------
# Evaluate MCTS
# -------------------------

iteration_list = [1, 5, 10, 25, 50, 100, 250, 500]

iters, avg_returns, return_sems, win_rates, avg_turns = PB.evaluate_mcts_sweep(
    m;
    iteration_list = iteration_list,
    n_episodes = 100,
    depth = 10,
    exploration_constant = 1.0,
    max_turns = 50
)

p1 = plot(
    iters,
    avg_returns,
    ribbon = return_sems,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Return",
    label = "Average Return ± SEM",
    title = "Average Return vs MCTS Search Budget"
)

savefig(p1, "mcts_avg_return_vs_iterations.png")

p2 = plot(
    iters,
    win_rates,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Win Rate",
    ylim = (0, 1.2),
    label = "Win Rate",
    title = "Win Rate vs MCTS Search Budget"
)

savefig(p2, "mcts_win_rate_vs_iterations.png")

