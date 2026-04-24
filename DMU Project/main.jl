include("src/PokemonBattle.jl")
using Plots

const PB = Main.PokemonBattle

m = PokemonBattleMDP(
    0.95,         # discount
    ElectricType, # my type
    WaterType,    # opp type
    100,           # my max hp
    100,           # opp max hp
    1,            # my attack
    1             # opp attack
)

s0 = PokemonState(
    m.my_max_hp,
    m.opp_max_hp,
    0,
    0,
    0,
    0
)

returns, avg_return = PB.evaluate_mcts(m; n_episodes=100)

episodes = 1:length(returns)

p = plot(
    episodes,
    returns,
    xlabel = "Episode",
    ylabel = "Return",
    label = "Episode Return",
    title = "MCTS Returns Over Episodes"
)

hline!(p, [avg_return], label = "Average Return")

savefig(p, "mcts_returns.png")
println("Saved plot to mcts_returns.png")