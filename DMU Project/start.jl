using POMDPs
using POMDPTools
using POMDPTools: SparseCat
using MCTS
#using Distributions
using Random
using POMDPTools: weighted_iterator

struct PokemonState
    my_hp::Int
    opp_hp::Int
    my_status::Int
    opp_status::Int
    my_boost::Int
    opp_boost::Int
end

@enum PokemonAction begin
    tackle
    thunder_wave
    swords_dance
    recover
end

struct PokemonBattleMDP <: MDP{PokemonState, PokemonAction}
    discount_factor::Float64
end

function POMDPs.states(m::PokemonBattleMDP)
    states = PokemonState[]

    for my_hp in 0:10
        for opp_hp in 0:10
            for my_status in 0:1
                for opp_status in 0:1
                    for my_boost in -2:2
                        for opp_boost in -2:2
                            push!(states, PokemonState(
                                my_hp,
                                opp_hp,
                                my_status,
                                opp_status,
                                my_boost,
                                opp_boost
                            ))
                        end
                    end
                end
            end
        end
    end

    return states
end

POMDPs.actions(m::PokemonBattleMDP) = [
    tackle,
    thunder_wave,
    swords_dance,
    recover
]

function opponent_policy(s::PokemonState)
    return [
        (tackle, 0.7),
        (recover, 0.2),
        (thunder_wave, 0.1)
    ]
end

POMDPs.discount(m::PokemonBattleMDP) = m.discount_factor

function POMDPs.reward(m::PokemonBattleMDP, s::PokemonState, a::PokemonAction, sp::PokemonState)
    if sp.opp_hp <= 0
        return 100.0
    elseif sp.my_hp <= 0
        return -100.0
    else
        damage_dealt = s.opp_hp - sp.opp_hp
        damage_taken = s.my_hp - sp.my_hp
        return damage_dealt - damage_taken
    end
end


function POMDPs.transition(m::PokemonBattleMDP, s::PokemonState, a::PokemonAction)
    next_states = PokemonState[]
    probs = Float64[]

    opp_dist = opponent_policy(s)

    for (opp_action, p_opp) in opp_dist
        outcomes = resolve_turn(s, a, opp_action)

        for (p_turn, sp) in outcomes
            push!(next_states, sp)
            push!(probs, p_opp * p_turn)
        end
    end

    return SparseCat(next_states, probs)
end

POMDPs.isterminal(m::PokemonBattleMDP, s::PokemonState) = s.my_hp <= 0 || s.opp_hp <= 0


function apply_my_action(s::PokemonState, a::PokemonAction)
    if a == tackle
        return [
            (0.9, PokemonState(
                s.my_hp,
                max(s.opp_hp - 2, 0),
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            )),
            (0.1, s)
        ]

    elseif a == thunder_wave
        return [
            (1.0, PokemonState(
                s.my_hp,
                s.opp_hp,
                s.my_status,
                1,
                s.my_boost,
                s.opp_boost
            ))
        ]

    elseif a == swords_dance
        return [
            (1.0, PokemonState(
                s.my_hp,
                s.opp_hp,
                s.my_status,
                s.opp_status,
                min(s.my_boost + 1, 2),
                s.opp_boost
            ))
        ]

    elseif a == recover
        return [
            (1.0, PokemonState(
                min(s.my_hp + 2, 10),
                s.opp_hp,
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            ))
        ]
    end
end

function apply_opp_action(s::PokemonState, a::PokemonAction)
    if a == tackle
        return [
            (0.9, PokemonState(
                max(s.my_hp - 2, 0),
                s.opp_hp,
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            )),
            (0.1, s)
        ]

    elseif a == thunder_wave
        return [
            (1.0, PokemonState(
                s.my_hp,
                s.opp_hp,
                1,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            ))
        ]

    elseif a == swords_dance
        return [
            (1.0, PokemonState(
                s.my_hp,
                s.opp_hp,
                s.my_status,
                s.opp_status,
                s.my_boost,
                min(s.opp_boost + 1, 2)
            ))
        ]

    elseif a == recover
        return [
            (1.0, PokemonState(
                s.my_hp,
                min(s.opp_hp + 2, 10),
                s.my_status,
                s.opp_status,
                s.my_boost,
                s.opp_boost
            ))
        ]
    end
end

function resolve_turn(s::PokemonState, my_action::PokemonAction, opp_action::PokemonAction)
    outcomes = Tuple{Float64, PokemonState}[]

    my_outcomes = apply_my_action(s, my_action)

    for (p_my, s_after_me) in my_outcomes

        # If opponent fainted, turn ends immediately
        if s_after_me.opp_hp <= 0
            push!(outcomes, (p_my, s_after_me))
            continue
        end

        opp_outcomes = apply_opp_action(s_after_me, opp_action)

        for (p_opp_turn, sp) in opp_outcomes
            push!(outcomes, (p_my * p_opp_turn, sp))
        end
    end

    return outcomes
end

function sample_opponent_action(s::PokemonState, rng::AbstractRNG)
    dist = opponent_policy(s)
    x = rand(rng)
    cumulative = 0.0

    for (a, p) in dist
        cumulative += p
        if x <= cumulative
            return a
        end
    end

    # fallback because of floating point roundoff
    return first(keys(dist))
end

function sample_turn_outcome(outcomes::Vector{Tuple{Float64, PokemonState}}, rng::AbstractRNG)
    if isempty(outcomes)
        error("sample_turn_outcome received an empty outcome list.")
    end

    x = rand(rng)
    cumulative = 0.0

    for (p, sp) in outcomes
        cumulative += p
        if x <= cumulative
            return sp
        end
    end

    return outcomes[end][2]
end

function POMDPs.gen(m::PokemonBattleMDP, s::PokemonState, a::PokemonAction, rng::AbstractRNG)
    opp_action = sample_opponent_action(s, rng)

    outcomes = resolve_turn(s, a, opp_action)

    sp = sample_turn_outcome(outcomes, rng)

    r = reward(m, s, a, sp)

    return (sp=sp, r=r)
end

m = PokemonBattleMDP(0.95)

## TEST A SINGLE INSTANCE ## 
# s = PokemonState(10, 10, 0, 0, 0, 0)
# rng = MersenneTwister(1)

# result = gen(m, s, tackle, rng)

# println(result.sp)
# println(result.r)

# for i in 1:10
#     result = gen(m, s, tackle, rng)
#     println(result)
# end

## APPLY MCTS ##
solver = MCTSSolver(
    n_iterations = 100,
    depth = 100,
    exploration_constant = 1.0
)

planner = solve(solver, m)

s = PokemonState(10, 10, 0, 0, 0, 0)

a = action(planner, s)

println("MCTS chose action: ", a)

# ### GRAPHING MCTS INSTANCE ###
using POMDPTools: weighted_iterator
POMDPs.actions(m::PokemonBattleMDP, s::PokemonState) = actions(m)

function state_label(s::PokemonState)
    return "myHP=$(s.my_hp), oppHP=$(s.opp_hp)\nmyStatus=$(s.my_status), oppStatus=$(s.opp_status)\nmyBoost=$(s.my_boost), oppBoost=$(s.opp_boost)"
end

function dot_escape(x)
    return replace(string(x), "\"" => "\\\"")
end

function make_transition_graph(m::PokemonBattleMDP, start_state::PokemonState;
                               max_depth::Int=3,
                               filename::String="pokemon_transition_graph.dot")

    node_ids = Dict{PokemonState, String}()
    depths = Dict{PokemonState, Int}()
    lines = String[]

    push!(lines, "digraph PokemonMDP {")
    push!(lines, "rankdir=LR;")
    push!(lines, "node [shape=box];")

    node_counter = 0

    function get_node_id(s::PokemonState)
        if !haskey(node_ids, s)
            node_counter += 1
            id = "S$(node_counter)"
            node_ids[s] = id

            shape = isterminal(m, s) ? "doublecircle" : "box"
            label = dot_escape(state_label(s))

            push!(lines, "$id [label=\"$label\", shape=$shape];")
        end

        return node_ids[s]
    end

    queue = PokemonState[start_state]
    depths[start_state] = 0
    get_node_id(start_state)

    while !isempty(queue)
        s = popfirst!(queue)
        current_depth = depths[s]

        if current_depth >= max_depth || isterminal(m, s)
            continue
        end

        s_id = get_node_id(s)

        for a in actions(m, s)
            d = transition(m, s, a)

            for (sp, p) in weighted_iterator(d)
                if p <= 0.0
                    continue
                end

                sp_id = get_node_id(sp)
                r = reward(m, s, a, sp)

                edge_label = "$(a)\\np=$(round(p, digits=2)), r=$(round(r, digits=2))"
                push!(lines, "$s_id -> $sp_id [label=\"$(dot_escape(edge_label))\"];")

                if !haskey(depths, sp)
                    depths[sp] = current_depth + 1
                    push!(queue, sp)
                end
            end
        end
    end

    push!(lines, "}")

    open(filename, "w") do io
        write(io, join(lines, "\n"))
    end

    println("Wrote graph to: ", filename)
end
s0 = PokemonState(10, 10, 0, 0, 0, 0)
make_transition_graph(m, s0, max_depth=2)
using Graphviz_jll
run(`$(Graphviz_jll.dot()) -Tpng pokemon_transition_graph.dot -o pokemon_transition_graph.png`)

### RUNNING FULL GAME ###
function play_mcts_game(m::PokemonBattleMDP;
                        start_state::PokemonState = PokemonState(10, 10, 0, 0, 0, 0),
                        max_turns::Int = 50,
                        seed::Int = 1)

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

    println("Starting game")
    println("Initial state: ", s)
    println()

    while !isterminal(m, s) && turn <= max_turns
        # MCTS reruns online search from the current state
        a = action(planner, s)

        # Environment completes one full turn
        result = gen(m, s, a, rng)

        sp = result.sp
        r = result.r

        println("Turn ", turn)
        println("Current state: ", s)
        println("MCTS chose: ", a)
        println("Next state: ", sp)
        println("Reward: ", r)
        println()

        s = sp
        total_reward += r
        turn += 1
    end

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

    return s, total_reward
end

m = PokemonBattleMDP(0.95)

final_state, total_reward = play_mcts_game(m)