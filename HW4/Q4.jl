import Cairo
import Fontconfig
using DMUStudent.HW4: HW4
using CommonRLInterface: act!, observe, reset!, terminated, render
using Statistics: mean
using Plots
using Random
using StaticArrays: SA

const ACTION_LIST = [
    SA[1, 0],
    SA[-1, 0],
    SA[0, 1],
    SA[0, -1]
]

function q_learning_episode!(Q, env; ϵ, γ, α, max_steps)
    start = time()

    function policy(s)
        if rand() < ϵ
            return rand(ACTION_LIST)
        else
            return argmax(a -> get(Q, (s, a), 0.0), ACTION_LIST)
        end
    end

    reset!(env)
    s = observe(env)
    hist = [s]

    t = 0
    while !terminated(env) && t < max_steps
        a = policy(s)
        r = act!(env, a)
        sp = observe(env)

        max_next_q = terminated(env) ? 0.0 : maximum(get(Q, (sp, ap), 0.0) for ap in ACTION_LIST)
        Q[(s, a)] = get(Q, (s, a), 0.0) + α * (r + γ * max_next_q - get(Q, (s, a), 0.0))

        s = sp
        push!(hist, s)
        t += 1
    end

    return (hist=hist, Q=copy(Q), time=time()-start, steps=t)
end

function q_learning!(env; n_episodes, α, γ, max_steps)
    Q = Dict{Tuple, Float64}()
    episodes = []

    for i in 1:n_episodes
        ϵ = max(0.1, 1 - i/(2*n_episodes))
        push!(episodes, q_learning_episode!(Q, env; ϵ=ϵ, γ=γ, α=α, max_steps=max_steps))

        # if i % 100 == 0
        #     println("finished episode $i")
        # end
    end

    return episodes
end

function greedy_policy(Q)
    return s -> argmax(a -> get(Q, (s, a), 0.0), ACTION_LIST)
end

function evaluate(env, policy; n_episodes=1000, γ=1.0, max_steps=15000)
    returns = Float64[]

    for _ in 1:n_episodes
        reset!(env)
        s = observe(env)
        r_total = 0.0
        t = 0

        while !terminated(env) && t < max_steps
            a = policy(s)
            r_total += γ^t * act!(env, a)
            s = observe(env)
            t += 1
        end

        push!(returns, r_total)
    end

    return returns
end

# function plot_value_grid(Q)
#     states_seen = [k[1] for k in keys(Q)]
#     nrows = maximum(s[1] for s in states_seen)
#     ncols = maximum(s[2] for s in states_seen)

#     V = fill(NaN, nrows, ncols)

#     for s in unique(states_seen)
#         V[s[1], s[2]] = maximum(get(Q, (s, a), 0.0) for a in ACTION_LIST)
#     end

#     heatmap(
#         1:ncols, 1:nrows, V;
#         yflip=true,
#         aspect_ratio=1,
#         xlabel="Column",
#         ylabel="Row",
#         title="Max Q-value per visited state",
#         colorbar=true
#     )
# end

function learning_curve_q_steps(q_episodes, env, SARSA_episodes)
    p = plot(
        xlabel="steps taken in environment",
        ylabel="avg return",
        title="Q-learning learning curve by step"
    )

    xs_q = Int[]
    ys_q = Float64[]
    xs_SARSA = Int[]
    ys_SARSA = Float64[]

    total_steps_q = 0
    total_steps_sarsa = 0

    for i in 1:length(SARSA_episodes)
        total_steps_sarsa += SARSA_episodes[i].steps

        if i % 100 == 0 || i < 1000
            Q = SARSA_episodes[i].Q
            π = greedy_policy(Q)
            avg_ret = mean(evaluate(env, π, n_episodes=100))
            push!(xs_SARSA, total_steps_sarsa)
            push!(ys_SARSA, avg_ret)
        end
    end

    for i in 1:length(q_episodes)
        total_steps_q += q_episodes[i].steps

        if i % 100 == 0 || i < 1000
            Q = q_episodes[i].Q
            π = greedy_policy(Q)
            avg_ret = mean(evaluate(env, π, n_episodes=100))
            push!(xs_q, total_steps_q)
            push!(ys_q, avg_ret)
        end
    end

    plot!(p, xs_q, ys_q, label="Q-learning", lw=1.5)
    plot!(p, xs_SARSA, ys_SARSA, label="SARSA", lw=1.5)
    return p
end

function learning_curve_q_time(q_episodes, env, SARSA_episodes)
    p = plot(
        xlabel="cumulative wall-clock time for training (s)",
        ylabel="avg return",
        title="Q-learning learning curve by time"
    )

    xs_q = Float64[]
    ys_q = Float64[]
    xs_SARSA = Float64[]
    ys_SARSA = Float64[]
    total_time_q = 0.0
    total_time_SARSA = 0.0

    for i in 1:length(q_episodes)
        total_time_q += q_episodes[i].time

        if i % 100 == 0 || i < 1000
            Q = q_episodes[i].Q
            π = greedy_policy(Q)
            avg_ret = mean(evaluate(env, π, n_episodes=100, γ=1.0))
            push!(xs_q, total_time_q)
            push!(ys_q, avg_ret)
        end
    end

    for i in 1:length(SARSA_episodes)
        total_time_SARSA += SARSA_episodes[i].time

        if i % 100 == 0 || i < 1000
            Q = SARSA_episodes[i].Q
            π = greedy_policy(Q)
            avg_ret = mean(evaluate(env, π, n_episodes=100, γ=1.0))
            push!(xs_SARSA, total_time_SARSA)
            push!(ys_SARSA, avg_ret)
        end
    end

    plot!(p, xs_q, ys_q, label="Q-learning", lw=1.5)
    plot!(p, xs_SARSA, ys_SARSA, label="SARSA", lw=1.5)
    return p
end

function SARSA_episode!(Q_SARSA, env; ϵ, γ, α, max_steps)
    start = time()

    function policy(s)
        if rand() < ϵ
            return rand(ACTION_LIST)
        else
            return argmax(a -> get(Q_SARSA, (s, a), 0.0), ACTION_LIST)
        end
    end

    reset!(env)
    s = observe(env)
    hist = [s]

    a = policy(s)   # choose first action before loop
    t = 0

    while !terminated(env) && t < max_steps
        r = act!(env, a)
        sp = observe(env)

        if terminated(env)
            target = r
        else
            ap = policy(sp)
            target = r + γ * get(Q_SARSA, (sp, ap), 0.0)
        end

        Q_SARSA[(s, a)] = get(Q_SARSA, (s, a), 0.0) + α * (target - get(Q_SARSA, (s, a), 0.0))

        if terminated(env)
            s = sp
        else
            s = sp
            a = ap
        end

        push!(hist, s)
        t += 1
    end

    return (hist=hist, Q=copy(Q_SARSA), time=time()-start, steps=t)
end

function SARSA_learning!(env; n_episodes, α, γ, max_steps)
    Q_SARSA = Dict{Tuple, Float64}()
    episodes = []

    for i in 1:n_episodes
        ϵ = max(0.1, 1 - i/(2*n_episodes))
        push!(episodes, SARSA_episode!(Q_SARSA, env; ϵ=ϵ, γ=γ, α=α, max_steps=max_steps))

        # if i % 100 == 0
        #     println("finished episode $i")
        # end
    end

    return episodes
end

env = HW4.gw

q_episodes = q_learning!(env, n_episodes=100000, α=0.04, γ=1.0, max_steps=15000)

Q_final = q_episodes[end].Q
π = greedy_policy(Q_final)

returns = evaluate(env, π, n_episodes=1000, γ=1.0, max_steps=15000)
println("mean return Q-Learning = ", mean(returns))

# display(plot_value_grid(Q_final))

# SARSA
SARSA_episodes = SARSA_learning!(env, n_episodes=100000, α=0.04, γ=1.0, max_steps=15000)
Q_final_SARSA = SARSA_episodes[end].Q
π = greedy_policy(Q_final_SARSA)
returns_SARSA = evaluate(env, π, n_episodes=1000, γ=1.0, max_steps=15000)
println("mean return SARSA = ", mean(returns_SARSA))

display(learning_curve_q_steps(q_episodes, env, SARSA_episodes))
display(learning_curve_q_time(q_episodes, env, SARSA_episodes))
#HW4.render(env)

# reset!(env)
# s0 = observe(env)
# println("start state = ", s0)

# for a in ACTION_LIST
#     reset!(env)
#     s = observe(env)
#     r = act!(env, a)
#     sp = observe(env)
#     println("action = $a : $s -> $sp, reward = $r")
# end