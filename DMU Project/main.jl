include("src/PokemonBattle.jl")
using .PokemonBattle
using Plots

const PB = PokemonBattle

# -------------------------
# Moves
# -------------------------

tackle = PB.Move(:tackle, :NormalType, :damage, 40, 1.0, :none)

thunderbolt = PB.Move(:thunderbolt, :ElectricType, :damage, 100, 1.0, :none)
nuzzle = PB.Move(:nuzzle, :ElectricType, :damage, 40, 1.0, :none)
grass_knot = PB.Move(:grass_knot, :GrassType, :damage, 120, 0.8, :none)

water_gun = PB.Move(:water_gun, :WaterType, :damage, 60, 1.0, :none)
bite = PB.Move(:bite, :DarkType, :damage, 100, 1.0, :none)

vine_whip = PB.Move(:vine_whip, :GrassType, :damage, 70, 1.0, :none)
razor_leaf = PB.Move(:razor_leaf, :GrassType, :damage, 90, 0.95, :none)

ember = PB.Move(:ember, :FireType, :damage, 60, 1.0, :none)
flame_burst = PB.Move(:flame_burst, :FireType, :damage, 90, 0.9, :none)
mud_slap = PB.Move(:mud_slap, :GroundType, :damage, 70, 1.0, :none)

# -------------------------
# My Team
# -------------------------

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

bulbasaur = PB.Pokemon(
    :Bulbasaur,
    :GrassType,
    1000,
    Dict(
        :move1 => tackle,
        :move2 => vine_whip,
        :move3 => razor_leaf,
        :move4 => mud_slap
    )
)

# -------------------------
# Opponent Team
# -------------------------

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

charmander = PB.Pokemon(
    :Charmander,
    :FireType,
    1000,
    Dict(
        :move1 => tackle,
        :move2 => ember,
        :move3 => flame_burst,
    )
)

# -------------------------
# MDP: 2v2 Team Battle
# -------------------------

my_team = [bulbasaur, charmander, squirtle, pikachu]
opp_team = [bulbasaur, charmander, squirtle, pikachu]

m = PB.PokemonBattleMDP(
    my_team,
    opp_team,
    0.95
)

s0 = PB.PokemonState(
    [p.max_hp for p in m.my_team],
    [p.max_hp for p in m.opp_team],
    fill(:none, length(m.my_team)),
    fill(:none, length(m.opp_team)),
    1,  # my active Pokemon: Pikachu
    1,  # opponent active Pokemon: Squirtle
    0,
    0
)

# # -------------------------
# # Run one verbose game first
# # -------------------------

# final_state, total_reward = PB.play_mcts_game(
#     m;
#     start_state = s0,
#     max_turns = 100,
#     seed = 1,
#     verbose = true,
#     n_iterations = 5000,
#     depth = 60,
#     exploration_constant = 100.0
# )

# println("Final state from test game: ", final_state)
# println("Total reward from test game: ", total_reward)

iteration_list = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000]

iters, avg_returns, return_sems, win_rates, avg_turns, avg_switches, avg_runtimes =
    PB.evaluate_mcts_sweep(
        m;
        iteration_list = iteration_list,
        n_episodes = 10,
        depth = 50,
        exploration_constant = 100.0,
        max_turns = 150
    )

# Average return
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
savefig(p1, "mcts_avg_return_vs_iterations_4v4.png")

# Win rate
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
savefig(p2, "mcts_win_rate_vs_iterations_4v4.png")

# Average turns
p3 = plot(
    iters,
    avg_turns,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Turns per Battle",
    label = "Average Turns",
    title = "Battle Length vs MCTS Search Budget"
)
savefig(p3, "mcts_avg_turns_vs_iterations_4v4.png")

# Average switches
p4 = plot(
    iters,
    avg_switches,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Switches per Battle",
    label = "Average Switches",
    title = "Switching Behavior vs MCTS Search Budget"
)
savefig(p4, "mcts_switches_vs_iterations_4v4.png")

# Runtime
p5 = plot(
    iters,
    avg_runtimes,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Runtime per Battle [s]",
    label = "Runtime",
    title = "Runtime vs MCTS Search Budget"
)
savefig(p5, "mcts_runtime_vs_iterations_4v4.png")

println("Saved 4v4 MCTS performance plots.")
