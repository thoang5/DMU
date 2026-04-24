using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW3 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

Please make sure to update DMUStudent to gain access to the HW3 module.

=#

############
# Question 3
############

m = HW3.DenseGridWorld(seed=3)


function rollout(mdp, policy_function, s0, max_steps=100)
    r_total = 0.0
    t = 0
    s = s0

    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s)
        # @info("Step:", s, a)
        s, r = @gen(:sp, :r)(mdp, s, a)
        r_total += discount(mdp)^t * r
        t += 1
    end

    return r_total
end

function rollout2(mdp, policy_function, s0, max_steps=100)
    r_total = 0.0
    t = 0
    s = s0

    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s)
        # @info("Step:", s, a)
        s, r = @gen(:sp, :r)(mdp, s, a)
        r_total += discount(mdp)^t * r
        t += 1
    end

    return r_total
end

function heuristic_policy100b100(mdp, s)
    node = [20, 40, 60, 80, 100]

    target_x = node[argmin(abs.(node .- s[1]))]
    target_y = node[argmin(abs.(node .- s[2]))]

    dx = target_x - s[1]
    dy = target_y - s[2]

    if abs(dx) > abs(dy)
        return dx > 0 ? :right : :left
    else
        return dy > 0 ? :up : :down
    end
end

function uniform_random_policy(mdp, s)
    rand(actions(mdp))
end


function heuristic_policy(mdp, s)
    node = [20, 40, 60]

    target_x = node[argmin(abs.(node .- s[1]))]
    target_y = node[argmin(abs.(node .- s[2]))]

    dx = target_x - s[1]
    dy = target_y - s[2]

    if abs(dx) > abs(dy)
        return dx > 0 ? :right : :left
    else
        return dy > 0 ? :up : :down
    end
end

# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
@show results1 = [rollout(m, uniform_random_policy, rand(initialstate(m)), 100) for _ in 1:500]
print("\n")
mean_estimate1 = mean(results1)
std_estimate1 = std(results1)
sem1 = std_estimate1 / sqrt(length(results1))
# # @show results = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:10]

println(" ==== Uniform Random Policy Stats === ")
println("Mean: ", mean_estimate1)
println("Standard Error of the Mean (SEM): ", sem1)

@show results2 = [rollout(m, heuristic_policy, rand(initialstate(m)), 1000) for _ in 1:100]
mean_estimate2 = mean(results2)
std_estimate2 = std(results2)
sem2 = std_estimate2 / sqrt(length(results2))

println(" ==== Heuristic Policy Stats === ")
println("Mean: ", mean_estimate2)
println("Standard Error of the Mean (SEM): ", sem2)

println("Improvement: ", abs(mean_estimate1) - abs(mean_estimate2))
println("QUESTION 3 DONE")

############
# Question 4
############

m = HW3.DenseGridWorld(seed=4)

S = statetype(m)
A = actiontype(m)

# These would be appropriate containers for your Q, N, and t dictionaries:
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()

# This is an example state - it is a StaticArrays.SVector{2, Int}
s = SA[19,19]
@show typeof(s)
@assert s isa statetype(m)

# here is an example of how to visualize a dummy tree (q, n, and t should actually be filled in your mcts code, but for this we fill it manually)
# q[(SA[1,1], :right)] = 0.0
# q[(SA[2,1], :right)] = 0.0
# n[(SA[1,1], :right)] = 1
# n[(SA[2,1], :right)] = 0
# t[(SA[1,1], :right, SA[2,1])] = 1
# using Random

# ---------------------------------------
# Entry point: π(s)
# ---------------------------------------
function wrapper(mdp, s, d, q, n, t, m; c=100)
    for k in 1:m
        simulate!(mdp, s, d, q, n, t; c=100)
    end

    # return action with highest Q(s,a)
    return argmax(a -> get(q, (s,a), 0.0), actions(mdp))
end

# ---------------------------------------
# Simulate
# ---------------------------------------
function simulate!(mdp, s, d, q, n, t; c=100.0)

    if d <= 0
        return 0.0
    end
    
    γ = discount(mdp)

    # ---------------------------------------
    # Expansion
    # ---------------------------------------
    if !any(haskey(n, (s,a)) for a in actions(mdp))
        for a in actions(mdp)
            n[(s,a)] = 0
            q[(s,a)] = 0.0
        end
        return rollout(mdp, heuristic_policy, s, 1000)
    end

    # ---------------------------------------
    # Selection (UCB)
    # ---------------------------------------
    Ns = sum(get(n,(s,ap),0) for ap in actions(mdp))
    a = argmax(ap ->
        get(q,(s,ap),0.0) + c * sqrt(log(max(Ns,1)) / max(get(n,(s,ap),0),1)),
        actions(mdp)
    )

    # ---------------------------------------
    # Sample transition
    # ---------------------------------------
    sp, r = @gen(:sp, :r)(mdp, s, a)

    # Track transition count
    t[(s,a,sp)] = get(t, (s,a,sp), 0) + 1

    # ---------------------------------------
    # Recursive simulation
    # ---------------------------------------
    G = float(r) + γ * simulate!(mdp, sp, d-1, q, n, t; c=c)

    # ---------------------------------------
    # Backup
    # ---------------------------------------
    n[(s,a)] = get(n,(s,a),0) + 1
    q_old = get(q,(s,a),0.0)
    q[(s,a)] = q_old + (G - q_old) / n[(s,a)]

    return G
end

function simulate!2(mdp, s, d, q, n, t; c=100.0)

    if d <= 0
        return 0.0
    end
    
    γ = discount(mdp)

    # ---------------------------------------
    # Expansion
    # ---------------------------------------
    if !any(haskey(n, (s,a)) for a in actions(mdp))
        for a in actions(mdp)
            n[(s,a)] = 0
            q[(s,a)] = 0.0
        end
        return rollout2(mdp, heuristic_policy100b100, s, 1000)
    end

    # ---------------------------------------
    # Selection (UCB / UCT)
    # ---------------------------------------
    Ns = sum(get(n, (s, ap), 0) for ap in actions(mdp))

    a = argmax(ap -> begin
            nsa = get(n, (s, ap), 0)
            qsa = get(q, (s, ap), 0.0)

            # Force trying each action at least once
            nsa == 0 ? Inf : qsa + c * sqrt(log(Ns + 1) / nsa)
        end,
        actions(mdp)
    )


    # ---------------------------------------
    # Sample transition
    # ---------------------------------------
    sp, r = @gen(:sp, :r)(mdp, s, a)

    # Track transition count
    t[(s,a,sp)] = get(t, (s,a,sp), 0) + 1

    # ---------------------------------------
    # Recursive simulation
    # ---------------------------------------
    G = float(r) + γ * simulate!(mdp, sp, d-1, q, n, t; c=c)

    # ---------------------------------------
    # Backup
    # ---------------------------------------
    n[(s,a)] = get(n,(s,a),0) + 1
    q_old = get(q,(s,a),0.0)
    q[(s,a)] = q_old + (G - q_old) / n[(s,a)]

    return G
end


m_iters = 7
d = 50
best_a = wrapper(m, s, d, q, n, t, m_iters; c=100.0)

inchrome(visualize_tree(q, n, t, s)) # use inbrowser(visualize_tree(q, n, t, SA[1,1]), "firefox") etc. if you want to use a different browser

############
# Question 5
############

# A starting point for the MCTS select_action function (a policy) which can be used for Questions 4 and 5
function select_action1(m, s)

    start = time_ns()
    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
    t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()
    
    # while time_ns() < start + 10_000_000
    for _ in 1:1000
    # while time_ns() < start + 40_000_000 # you can replace the above line with this if you want to limit this loop to run within 40ms
        d = 20
        simulate!(m, s, d, q, n, t; c=100.0)
        # break # replace this with mcts iterations to fill n and q
    end

    # select a good action based on q and/or n

    return argmax(a -> get(q, (s,a), -Inf), actions(m)) # this dummy function returns a random action, but you should return your selected action
end

function select_action2(m, s)

    start = time_ns()
    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
    t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()
    
    while time_ns() < start + 45_000_000
    #for _ in 1:1000
    # while time_ns() < start + 40_000_000 # you can replace the above line with this if you want to limit this loop to run within 40ms
        d = 40
        simulate!2(m, s, d, q, n, t; c=100.0)
        # break # replace this with mcts iterations to fill n and q
    end

    # select a good action based on q and/or n

    # for a in actions(m)
    #     println(a, "  n=", get(n,(s,a),0), "  q=", get(q,(s,a),0.0))
    # end

    return argmax(a -> get(q, (s,a), -Inf), actions(m)) # changed to N than q 
 # this dummy function returns a random action, but you should return your selected action
end

@btime select_action1(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.

# use the code below to evaluate the MCTS policy
#@show results3 = [rollout(m, select_action, rand(initialstate(m)), 100) for _ in 1:100]
results3 = [rollout(m, select_action1, rand(initialstate(m)), 100) for _ in 1:100]

mean_estimate3 = mean(results3)
std_estimate3 = std(results3)
sem3 = std_estimate3 / sqrt(length(results3))
println(" ==== MCTS Policy Stats === ")
println("Mean: ", mean_estimate3)
println("Standard Error of the Mean (SEM): ", sem3)

############
# Question 6
############

# @btime select_action1(m, SA[35,35])

HW3.evaluate(select_action2, "thomas.hoang@colorado.edu", time=true)

# # If you want to see roughly what's in the evaluate function (with the timing code removed), check sanitized_evaluate.jl

# ########
# # Extras
# ########

# # With a typical consumer operating system like Windows, OSX, or Linux, it is nearly impossible to ensure that your function *always* returns within 50ms. Do not worry if you get a few warnings about time exceeded.

# # You may wish to call select_action once or twice before submitting it to evaluate to make sure that all parts of the function are precompiled.

# # Instead of submitting a select_action function, you can alternatively submit a POMDPs.Solver object that will get 50ms of time to run solve(solver, m) to produce a POMDPs.Policy object that will be used for planning for each grid world.
