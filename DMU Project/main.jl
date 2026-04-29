include("src/PokemonBattle.jl")
using .PokemonBattle
using Plots

const PB = PokemonBattle

# -------------------------
# Output folder
# -------------------------

outdir = "Plots/Greedy Comparison"
mkpath(outdir)

# -------------------------
# Moves
# -------------------------

tackle = PB.Move(:tackle, :NormalType, :damage, 40, 1.0, :none)

thunderbolt = PB.Move(:thunderbolt, :ElectricType, :damage, 100, 1.0, :none)
thunder_wave = PB.Move(:thunder_wave, :ElectricType, :status, 0, 1.0, :paralyze)
grass_knot = PB.Move(:grass_knot, :GrassType, :damage, 120, 0.8, :none)

water_gun = PB.Move(:water_gun, :WaterType, :damage, 60, 1.0, :none)
bite = PB.Move(:bite, :DarkType, :damage, 100, 1.0, :none)

vine_whip = PB.Move(:vine_whip, :GrassType, :damage, 70, 1.0, :none)
razor_leaf = PB.Move(:razor_leaf, :GrassType, :damage, 90, 0.95, :none)

ember = PB.Move(:ember, :FireType, :damage, 60, 1.0, :none)
flame_burst = PB.Move(:flame_burst, :FireType, :damage, 90, 0.9, :none)

mud_slap = PB.Move(:mud_slap, :GroundType, :damage, 70, 1.0, :none)

swords_dance = PB.Move(:swords_dance, :NormalType, :boost, 0, 1.0, :attack_boost)
recover = PB.Move(:recover, :NormalType, :heal, 40, 1.0, :recover)

# -------------------------
# Pokemon
# Format:
# Pokemon(name, type, max_hp, speed, moves)
# -------------------------

pikachu = PB.Pokemon(
    :Pikachu,
    :ElectricType,
    1000,
    100,
    Dict(
        :move1 => tackle,
        :move2 => thunderbolt,
        :move3 => thunder_wave,
        :move4 => grass_knot
    )
)

bulbasaur = PB.Pokemon(
    :Bulbasaur,
    :GrassType,
    1000,
    45,
    Dict(
        :move1 => recover,
        :move2 => vine_whip,
        :move3 => razor_leaf,
        :move4 => mud_slap
    )
)

squirtle = PB.Pokemon(
    :Squirtle,
    :WaterType,
    1000,
    50,
    Dict(
        :move1 => tackle,
        :move2 => water_gun,
        :move3 => bite,
        :move4 => swords_dance
    )
)

charmander = PB.Pokemon(
    :Charmander,
    :FireType,
    1000,
    60,
    Dict(
        :move1 => tackle,
        :move2 => ember,
        :move3 => flame_burst,
        :move4 => swords_dance
    )
)

# -------------------------
# MDP: 4v4 Singles Battle
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
    1,  # my active Pokemon
    1,  # opponent active Pokemon
    0,
    0
)

# -------------------------
# MCTS Sweep Settings
# -------------------------

iteration_list = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
#iteration_list = [1, 5]

n_episodes = 100
depth = 50
exploration_constant = 500.0
max_turns = 500

# -------------------------
# Evaluate MCTS Sweep
# -------------------------

iters, avg_returns, return_sems, win_rates, avg_turns, avg_switches, avg_runtimes =
    PB.evaluate_mcts_sweep(
        m;
        iteration_list = iteration_list,
        n_episodes = n_episodes,
        depth = depth,
        exploration_constant = exploration_constant,
        max_turns = max_turns
    )

# -------------------------
# Evaluate Greedy Baseline
# -------------------------

greedy_avg_return, greedy_win_rate, greedy_avg_turns, greedy_avg_switches =
    PB.evaluate_greedy(
        m;
        n_episodes = n_episodes,
        max_turns = max_turns
    )

println()
println("Greedy baseline results:")
println("Average return: ", greedy_avg_return)
println("Win rate: ", greedy_win_rate)
println("Average turns: ", greedy_avg_turns)
println("Average switches: ", greedy_avg_switches)

# -------------------------
# Plot 1: MCTS Average Return
# -------------------------

p1 = plot(
    iters,
    avg_returns,
    ribbon = return_sems,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Return",
    label = "MCTS ± SEM",
    title = "Average Return vs MCTS Search Budget"
)

savefig(p1, joinpath(outdir, "mcts_avg_return_vs_iterations_4v4.png"))
println("Saved mcts_avg_return_vs_iterations_4v4.png")

# -------------------------
# Plot 2: MCTS Win Rate
# -------------------------

p2 = plot(
    iters,
    win_rates,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Win Rate",
    ylim = (0, 1.2),
    label = "MCTS",
    title = "Win Rate vs MCTS Search Budget"
)

savefig(p2, joinpath(outdir, "mcts_win_rate_vs_iterations_4v4.png"))
println("Saved mcts_win_rate_vs_iterations_4v4.png")

# -------------------------
# Plot 3: MCTS Average Turns
# -------------------------

p3 = plot(
    iters,
    avg_turns,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Turns per Battle",
    label = "MCTS",
    title = "Battle Length vs MCTS Search Budget"
)

savefig(p3, joinpath(outdir, "mcts_avg_turns_vs_iterations_4v4.png"))
println("Saved mcts_avg_turns_vs_iterations_4v4.png")

# -------------------------
# Plot 4: MCTS Average Switches
# -------------------------

p4 = plot(
    iters,
    avg_switches,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Switches per Battle",
    label = "MCTS",
    title = "Switching Behavior vs MCTS Search Budget"
)

savefig(p4, joinpath(outdir, "mcts_switches_vs_iterations_4v4.png"))
println("Saved mcts_switches_vs_iterations_4v4.png")

# -------------------------
# Plot 5: MCTS Runtime
# -------------------------

p5 = plot(
    iters,
    avg_runtimes,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Runtime per Battle [s]",
    label = "MCTS",
    title = "Runtime vs MCTS Search Budget"
)

savefig(p5, joinpath(outdir, "mcts_runtime_vs_iterations_4v4.png"))
println("Saved mcts_runtime_vs_iterations_4v4.png")

# -------------------------
# Plot 6: MCTS vs Greedy Win Rate
# -------------------------

p6 = plot(
    iters,
    win_rates,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Win Rate",
    ylim = (0, 1.2),
    label = "MCTS",
    title = "MCTS vs Greedy Heuristic Win Rate"
)

hline!(p6, [greedy_win_rate], label = "Greedy Heuristic")

savefig(p6, joinpath(outdir, "mcts_vs_greedy_win_rate_4v4.png"))
println("Saved mcts_vs_greedy_win_rate_4v4.png")

# -------------------------
# Plot 7: MCTS vs Greedy Average Return
# -------------------------

p7 = plot(
    iters,
    avg_returns,
    ribbon = return_sems,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Return",
    label = "MCTS ± SEM",
    title = "MCTS vs Greedy Heuristic Average Return"
)

hline!(p7, [greedy_avg_return], label = "Greedy Heuristic")

savefig(p7, joinpath(outdir, "mcts_vs_greedy_avg_return_4v4.png"))
println("Saved mcts_vs_greedy_avg_return_4v4.png")

# -------------------------
# Plot 8: MCTS vs Greedy Average Turns
# -------------------------

p8 = plot(
    iters,
    avg_turns,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Turns per Battle",
    label = "MCTS",
    title = "MCTS vs Greedy Heuristic Battle Length"
)

hline!(p8, [greedy_avg_turns], label = "Greedy Heuristic")

savefig(p8, joinpath(outdir, "mcts_vs_greedy_avg_turns_4v4.png"))
println("Saved mcts_vs_greedy_avg_turns_4v4.png")

# -------------------------
# Plot 9: MCTS vs Greedy Average Switches
# -------------------------

p9 = plot(
    iters,
    avg_switches,
    marker = :circle,
    xscale = :log10,
    xlabel = "MCTS Iterations per Move",
    ylabel = "Average Switches per Battle",
    label = "MCTS",
    title = "MCTS vs Greedy Heuristic Switching"
)

hline!(p9, [greedy_avg_switches], label = "Greedy Heuristic")

savefig(p9, joinpath(outdir, "mcts_vs_greedy_switches_4v4.png"))
println("Saved mcts_vs_greedy_switches_4v4.png")

println()
println("Finished MCTS sweep and greedy baseline comparison.")