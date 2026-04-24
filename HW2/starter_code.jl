using DMUStudent.HW2
using POMDPs: states, actions
using POMDPTools: ordered_states, render
import Cairo, Fontconfig # Needed in some cases for rendering the value function on grid world
using LinearAlgebra, SparseArrays, Base.Threads

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW2 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

=#

############
# Question 3
############

@show actions(grid_world) # prints the actions. In this case each action is a Symbol. Use ?Symbol to find out more.

T = transition_matrices(grid_world)
display(T) # this is a Dict that contains a transition matrix for each action

@show T[:left][1, 2] # the probability of transitioning between states with indices 1 and 2 when taking action :left

R = reward_vectors(grid_world)
display(R) # this is a Dict that contains a reward vector for each action

@show R[:right][1] # the reward for taking action :right in the state with index 1

function value_iteration_Q4(m)
    # It is good to put performance-critical code in a function: https://docs.julialang.org/en/v1/manual/performance-tips/
    S = collect(states(m))          # list of states (or just use indices)
    A = collect(actions(m))         # list of actions (Symbols)
    T = transition_matrices(m; sparse=true)      # Dict{Symbol, Matrix}
    R = reward_vectors(m)           # Dict{Symbol, Vector}
    gamma = 0.95

    V = zeros(length(states(m))) # this would be a good container to use for your value function
    V_prime = ones(length(states(m)))
    tol = 1e-6
    ns = length(states(m))
    while norm(V - V_prime, Inf) > tol

        V .= V_prime
        for s in 1:ns
            V_prime[s] = maximum(R[a][s] + gamma*dot(T[a][s,1:ns], V) for a in A)
        end
    end
    # put your value iteration code here

    return V_prime
end

function value_iteration_Q5(m; gamma=0.99, tol=1e-8, max_iter=10_000)
    A = collect(actions(m))
    T = transition_matrices(m; sparse=true)
    R = reward_vectors(m)

    ns = length(first(values(R)))
    V  = zeros(ns)
    Vp = similar(V)

    tmp = Dict(a => zeros(ns) for a in A)

    for it in 1:max_iter
        @threads for i in eachindex(A)
            a = A[i]
            mul!(tmp[a], T[a], V)
            @. tmp[a] = R[a] + gamma * tmp[a]
        end

        Vp .= tmp[A[1]]
        for a in A[2:end]
            Vp .= max.(Vp, tmp[a])
        end

        if norm(Vp - V, Inf) < tol
            return Vp
        end
        V .= Vp
    end

    return V
end

V = value_iteration_Q4(grid_world)
#V = rand(length(states(grid_world)))*10.0 # replace this with value_iteration(m)
# If you are in an environment with multimedia capability (e.g. VSCode, Jupyter, Pluto), use this:
display(render(grid_world, color=V)) # In the REPL, this will output an annoying amount of text
# If you are in the REPL or want to save a png, use this:
using Compose: draw, PNG
draw(PNG("value.png"), render(grid_world, color=V))

############
# Question 4
############

# You can create an mdp object representing the problem with the following:
#n = 7
m = UnresponsiveACASMDP(7)

# transition_matrices and reward_vectors work the same as for grid_world, however this problem is much larger, so you will have to exploit the structure of the problem. In particular, you may find the docstring of transition_matrices helpful:
# display(@doc(transition_matrices))

V = value_iteration_Q5(m)

@show HW2.evaluate(V, "thomas.hoang@colorado.edu")

########
# Extras
########

# The comments below are not needed for the homework, but may be helpful for interpreting the problems or getting a high score on the leaderboard.

# Both UnresponsiveACASMDP and grid_world implement the POMDPs.jl interface. You can find complete documentation here: https://juliapomdp.github.io/POMDPs.jl/stable/api/#Model-Functions

# To convert from physical states to indices in the transition function, use the stateindex function
# IMPORTANT NOTE: YOU ONLY NEED TO USE STATE INDICES FOR THIS ASSIGNMENT, using the states may help you make faster specialized code for the ACAS problem, but it is not required
using POMDPs: states, stateindex

s = first(states(m))
@show si = stateindex(m, s)

# To convert from a state index to a physical state in the ACAS MDP, use convert_s:
using POMDPs: convert_s

@show s = convert_s(ACASState, si, m)

# To visualize a state in the ACAS MDP, use
render(m, (s=s,))
