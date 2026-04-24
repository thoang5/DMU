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

function render_transition_graph(dotfile::String="pokemon_transition_graph.dot",
                                 pngfile::String="pokemon_transition_graph.png")
    run(`$(Graphviz_jll.dot()) -Tpng $dotfile -o $pngfile`)
    println("Wrote PNG to: ", pngfile)
end