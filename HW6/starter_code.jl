using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater, has_consistent_distributions
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
# using SARSOP: SARSOPSolver
using NativeSARSOP: SARSOPSolver, alphavectors

using LinearAlgebra
using Plots
using Statistics

##################
# Problem 1: Tiger
##################

#--------
# Updater
#--------

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    m = up.m
    bp_vec = zeros(length(states(m)))
    # bp_vec[1] = 1.0
    
    for sp in states(m)
        spi = stateindex(m,sp)

        pred = 0.0
        # getting the prediction states sum sum(P*b)
        for s in states(m)
            si = stateindex(m,s)
            pred += pdf(transition(m,s,a), sp) * b.b[si]
        end

        # multiply it by the observation transition matrix (Z) = b' = Z*sum(P*b)
        bp_vec[spi] = pdf(observation(m,a,sp), o) * pred
    end

    # normalization over belief update
    total = sum(bp_vec)
    if total == 0.0
        error("Yeah there wasn't SHIT in there")
    end

    bp_vec ./= total
    # Fill in code for belief update
    # Note that the ordering of the entries in bp_vec must be consistent with stateindex(m, s) (the container returned by states(m) does not necessarily obey this order)

    return DiscreteBelief(up.m, bp_vec)
end

# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example if you want to use more mathematical-looking functions, you could use the following:
# Z(o | a, s') can be programmed with
Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.

# This is needed to automatically turn any distribution into a discrete belief.
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

# Note: to check your belief updater code, you can use POMDPTools: DiscreteUpdater. It should function exactly like your updater.

# -------
# Policy
# -------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    # @assert !isempty(p.alphas) "Policy has no alpha vectors."
    # @assert !isempty(p.alpha_actions) "Policy has no alpha actions."
    # @assert length(p.alphas) == length(p.alpha_actions) "Number of alpha vectors must match number of alpha actions."
    # # Fill in code to choose action based on alpha vectors

    best_i = 1
    best_val = -Inf

    @assert length(p.alphas) == length(p.alpha_actions) "Number of alpha vectors must match number of alpha actions."

    # actions = max(alpha*belief)
    for i in eachindex(p.alphas)
        α = p.alphas[i]
        # @assert length(α) == length(b.b) "Alpha vector length must match belief length."

        v = dot(α, b.b)

        if v > best_val
            best_val = v
            best_i = i
        end
    end

    # return first(actions(b.pomdp))
    return p.alpha_actions[best_i]
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order

#------
# QMDP
#------

function qmdp_solve(m, γ=discount(m))

    # Fill in Value Iteration to compute the Q-values

    sts = ordered_states(m)
    nS = length(sts)

    # Value Iteration
    U = zeros(nS)
    U_new = zeros(nS)

    tol = 1e-8
    max_iters = 500

    for iter in 1:max_iters
        for s in sts
            si = stateindex(m,s)

            U_new[si] = maximum(
                begin
                    future = 0.0
                    for sp in sts
                        spi = stateindex(m,sp)
                        future += pdf(transition(m,s,a), sp) * U[spi]
                    end
                    reward(m,s,a) + γ*future
                end
                for a in actions(m)
            )
        end

        if maximum(abs.(U_new-U)) < tol
            U .= U_new
            break
        end
        U .= U_new
    end

    # link an alpha vector per action
    acts = actiontype(m)[]
    alphas = Vector{Float64}[]
    for a in actions(m)

        # Fill in pseudo alpha vector calculation
        # Note that the ordering of the entries in the pseudo alpha vectors must be consistent with stateindex(m, s) (states(m) does not necessarily obey this order, but ordered_states(m) does.)
        
        α = zeros(nS)

        for s in sts
            si = stateindex(m,s)

            future = 0.0
            for sp in sts
                spi = stateindex(m,sp)
                future += pdf(transition(m,s,a), sp) * U[spi]
            end
            
            α[si] = reward(m,s,a) + γ*future
        end
        push!(acts, a)
        push!(alphas, α)
    end
    return HW6AlphaVectorPolicy(alphas, acts)
end

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

@show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
@show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)

# Monte Carlo evaluation
N = 5000
sim = RolloutSimulator(max_steps=500)

qmdp_returns = [simulate(sim, m, qmdp_p, up) for _ in 1:N]
sarsop_returns = [simulate(sim, m, sarsop_p, up) for _ in 1:N]

qmdp_mean = mean(qmdp_returns)
qmdp_sem  = std(qmdp_returns) / sqrt(N)

sarsop_mean = mean(sarsop_returns)
sarsop_sem  = std(sarsop_returns) / sqrt(N)

@show qmdp_mean qmdp_sem
@show sarsop_mean sarsop_sem

# Extract alpha vectors
qmdp_alphas = qmdp_p.alphas
sarsop_alphas = alphavectors(sarsop_p)

# Belief grid: p = P(tiger left)
ps = range(0.0, 1.0; length=300)

function alpha_value_curve(alpha, ps)
    [dot(alpha, [p, 1.0-p]) for p in ps]
end

tiger_action_label(a) =
    a == TIGER_LISTEN     ? "listen" :
    a == TIGER_OPEN_LEFT  ? "open left" :
    a == TIGER_OPEN_RIGHT ? "open right" :
    string(a)

ps = range(0.0, 1.0; length=300)

plt = plot(
    xlabel = "Belief b(tiger left)",
    ylabel = "Alpha value",
    title = "TigerPOMDP - SARSOP vs QMDP alpha vectors",
    legend = :best
)

for (i, α) in enumerate(qmdp_p.alphas)
    a = qmdp_p.alpha_actions[i]
    plot!(plt, ps, alpha_value_curve(α, ps),
          label = "QMDP: $(tiger_action_label(a))",
          linestyle = :dash)
end

for (i, α) in enumerate(alphavectors(sarsop_p))
    a = sarsop_p.action_map[i]
    plot!(plt, ps, alpha_value_curve(α, ps),
          label = "SARSOP: $(tiger_action_label(a))")
end

# @show propertynames(sarsop_p)
# @show fieldnames(typeof(sarsop_p))
# dump(sarsop_p)

display(plt)

###################
# Problem 2: Cancer
###################

# cancer = QuickPOMDP(

#     # Fill in your actual code from last homework here

#     states = [:healthy, :in_situ, :invasive, :death],
#     actions = [:wait, :test, :treat],
#     observations = [true, false],
#     transition = (s, a) -> Deterministic(s),
#     observation = (a, sp) -> Deterministic(false),
#     reward = (s, a) -> 0.0,
#     discount = 0.99,
#     initialstate = Deterministic(:death),
#     isterminal = s->s==:death,


# )

cancer = QuickPOMDP(
    states = [:healthy, :in_situ, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [true, false],

    transition = (s, a) -> begin
        if s == :healthy
            SparseCat([:healthy, :in_situ], [0.98, 0.02])

        elseif s == :in_situ
            if a == :treat
                SparseCat([:healthy, :in_situ], [0.6, 0.4])
            else
                SparseCat([:in_situ, :invasive], [0.9, 0.1])
            end

        elseif s == :invasive
            if a == :treat
                SparseCat([:healthy, :invasive, :death], [0.2, 0.6, 0.2])
            else
                SparseCat([:invasive, :death], [0.4, 0.6])
            end

        else
            Deterministic(:death)
        end
    end,

    observation = (a, sp) -> begin
        if a == :test
            if sp == :healthy
                SparseCat([true, false], [0.05, 0.95])   # positive, negative
            elseif sp == :in_situ
                SparseCat([true, false], [0.8, 0.2])
            elseif sp == :invasive
                Deterministic(true)
            else
                Deterministic(false)
            end

        elseif a == :treat
            if sp == :in_situ || sp == :invasive
                Deterministic(true)
            else
                Deterministic(false)
            end

        else
            Deterministic(false)
        end
    end,

    reward = (s, a) -> begin
        if s == :death
            0.0
        elseif a == :wait
            1.0
        elseif a == :test
            0.8
        else
            0.1
        end
    end,

    discount = 0.99,
    initialstate = Deterministic(:healthy),
    isterminal = s -> s == :death
)

@assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up = HW6Updater(cancer)

heuristic = FunctionPolicy(function (b)

                               # Fill in your heuristic policy here
                               # Use pdf(b, s) to get the probability of a state

                                # return :wait
                                p_healthy  = pdf(b, :healthy)
                                p_insitu   = pdf(b, :in_situ)
                                p_invasive = pdf(b, :invasive)
                                p_death    = pdf(b, :death)

                                # p_cancer = p_insitu + p_invasive

                                if p_invasive > 0.80 || p_insitu > 0.20
                                    return :treat
                                elseif p_healthy > 0.99
                                    return :wait
                                else
                                    return :test
                                end
                           end
                          )

@show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
@show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
@show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79
@show std(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)/1000
@show std(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)/1000
@show std(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)/1000

#####################
# Problem 4: LaserTag
#####################

# Particle Monte Carlo Planning
# using BasicPOMCP

# function pomcp_solve(m)
#     solver = POMCPSolver(
#         tree_queries = 10,
#         c = 1.0,
#         default_action = first(actions(m)),
#         estimate_value = FORollout(FunctionPolicy(s ->rand(actions(m))))
#     )
#     return solve(solver,m)
# end

m = LaserTagPOMDP()

# pomcp_p = pomcp_solve(m)

#qmdp_p = qmdp_solve(m)
# up = DiscreteUpdater(m) # you may want to replace this with your updater to test it

# Use this version with only 100 episodes to check how well you are doing quickly
#@show HW6.evaluate((qmdp_p, up), n_episodes=100)
# @show HW6.evaluate((pomcp_p, up), n_episodes=100)

# A good approach to try is POMCP, implemented in the BasicPOMCP.jl package:
using BasicPOMCP
using POMDPTools: FunctionPolicy
using StaticArrays

manhattan(a, b) = abs(a[1] - b[1]) + abs(a[2] - b[2])

robot_pos(s) = s.robot
target_pos(s) = s.target

function legal_move_results(m, s)
    results = Tuple{Symbol, typeof(s)}[]
    for a in (:up, :down, :left, :right)
        sp = rand(transition(m, s, a))
        if robot_pos(sp) != robot_pos(s)
            push!(results, (a, sp))
        end
    end
    return results
end

function best_legal_move(m, s)
    t = target_pos(s)
    results = legal_move_results(m, s)

    if isempty(results)
        return :measure
    end

    best_a = results[1][1]
    best_d = typemax(Int)

    for (a, sp) in results
        d = manhattan(robot_pos(sp), t)
        if d < best_d
            best_d = d
            best_a = a
        end
    end

    return best_a
end

function pomcp_solve(m)
    rollout_policy = FunctionPolicy(s -> best_legal_move(m, s))

    solver = POMCPSolver(
        tree_queries = 2000,
        c = 1.0,
        max_depth = 20,
        estimate_value = FORollout(rollout_policy)
    )

    return solve(solver, m)
end

s = rand(initialstate(m))
@show s
@show typeof(s)
@show propertynames(s)
@show fieldnames(typeof(s))
dump(s)

s = rand(initialstate(m))
@show s

for a in actions(m)
    println("\nAction: ", a)
    for i in 1:5
        sp = rand(transition(m, s, a))
        @show sp
    end
end

s = rand(initialstate(m))   # or build one manually near a wall
@show s

for a in (:up, :down, :left, :right)
    sp = rand(transition(m, s, a))
    println(a, ": ", s.robot, " -> ", sp.robot)
end

s_test = LTState(
    SVector(5, 5),   # robot
    SVector(2, 2),   # target
    SVector(5, 6)    # wanderer
)

@show s_test
sp = rand(transition(m, s_test, :right))
@show sp
@show reward(m, s_test, :right)

s_target = LTState(
    SVector(5, 5),
    SVector(5, 6),
    SVector(2, 2)
)

sp = rand(transition(m, s_target, :right))
@show sp
@show reward(m, s_target, :right)

function test_action(m, s, a; N=10)
    println("\nState: ", s)
    println("Action: ", a)
    println("Immediate reward: ", reward(m, s, a))
    for i in 1:N
        sp = rand(transition(m, s, a))
        println("  -> ", sp)
    end
end

test_action(m, s_test, :right)
test_action(m, s_target, :right)

function test_step_reward(m, s, a)
    sp = rand(transition(m, s, a))
    println("s      = ", s)
    println("a      = ", a)
    println("sp     = ", sp)

    try
        println("reward(m, s, a, sp) = ", reward(m, s, a, sp))
    catch
        println("reward(m, s, a, sp) not defined")
    end

    try
        println("reward(m, s, a)     = ", reward(m, s, a))
    catch
        println("reward(m, s, a) not defined")
    end

    try
        println("isterminal(sp)      = ", isterminal(m, sp))
    catch
        println("isterminal not available")
    end
end

# robot moves DOWN onto TARGET
s_hit_target = LTState(
    SVector(5,5),
    SVector(6,5),
    SVector(2,2)
)

# robot moves DOWN onto WANDERER
s_hit_wanderer = LTState(
    SVector(5,5),
    SVector(2,2),
    SVector(6,5)
)

test_step_reward(m, s_hit_target, :down)
test_step_reward(m, s_hit_wanderer, :down)

pomcp_p = pomcp_solve(m)
up = DiscreteUpdater(m)
#@show HW6.evaluate((pomcp_p, up), n_episodes=25)
@show HW6.evaluate((pomcp_p, up), "thomas.hoang@colorado.edu")

# When you get ready to submit, use this version with the full 1000 episodes
# HW6.evaluate((qmdp_p, up), "REPLACE_WITH_YOUR_EMAIL@colorado.edu")

#----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
using POMDPGifs
import Cairo, Fontconfig # needed to display properly

#makegif(m, qmdp_p, up, max_steps=30, filename="lasertag.gif")
makegif(m, pomcp_p, up, max_steps=100, filename="lasertag.gif")

# You can render a single frame like this
using POMDPTools: stepthrough, render
using Compose: draw, PNG

history = []
for step in stepthrough(m, pomcp_p, up, max_steps=10)
    push!(history, step)
end
displayable_object = render(m, last(history))
# display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
draw(PNG("lasertag.png"), displayable_object)
