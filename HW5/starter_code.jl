using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW5 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

=#

############
# Question 1
############

# The tiger problem from http://www.sciencedirect.com/science/article/pii/S000437029800023X can be expressed with:

tiger = QuickPOMDP(
    states = [:TL, :TR],
    actions = [:OL, :OR, :L],
    observations = [:TL, :TR],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition = function (s, a)
        if a == :L
            return Deterministic(s)
        else
            return Uniform([:TL, :TR])
        end
    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation = function (s, a, sp)
        if a == :L
            if sp == :TL
                return SparseCat([:TL, :TR], [0.85, 0.15])
            else
                return SparseCat([:TR, :TL], [0.85, 0.15])
            end
        else
            return Uniform([:TL, :TR])
        end
    end,

    reward = function (s, a)
        if a == :L
            return -1.0
        elseif a == :OL
            if s == :TR
                return 10.0
            else
                return -100.0
            end
        else # a = :OR
            if s == :TL
                return 10.0
            else
                return -100.0
            end
        end
    end,

    initialstate = Uniform([:TL, :TR]),

    discount = 0.95
)

# evaluate with a random policy
policy = FunctionPolicy(o->rand(POMDPs.actions(tiger)))
sim = RolloutSimulator(max_steps=100)
@show @time mean(POMDPs.simulate(sim, tiger, policy) for _ in 1:10_000)

############
# Question 2
############

# The notebook at https://github.com/zsunberg/CU-DMU-Materials/blob/master/notebooks/110-Neural-Networks.jl can serve as a starting point for this problem.

cancer = QuickPOMDP(
    states = [:healthy, :insitu, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [:positive, :negative],

    transition = function(s,a)
        if s == :healthy
            return SparseCat([:healthy, :insitu], [0.98, 0.02])

        elseif s== :insitu
            if a ==:treat
                return SparseCat([:healthy, :insitu], [0.6, 0.4])
            else
                return SparseCat([:insitu, :invasive], [0.9, 0.1])
            end

        elseif s == :invasive
            if a ==:treat
                return SparseCat([:healthy, :invasive, :death], [0.2, 0.6, 0.2])
            else
                return SparseCat([:invasive, :death], [0.4, 0.6])
            end

        else # death
            return Deterministic(:death)
        end
    end, 

    observation = function(s,a,sp)
        if a == :test
            if sp == :healthy
                return SparseCat([:positive, :negative], [0.05, 0.95])
            elseif sp == :insitu
                return SparseCat([:positive, :negative], [0.8, 0.2])
            elseif sp == :invasive
                return Deterministic(:positive)
            else # death
                return Deterministic(:negative)
            end
            
        elseif a == :treat
            if sp == :insitu || sp == :invasive
                return Deterministic(:positive)
            else
                return Deterministic(:negative)
            end

        else # wait
            return Deterministic(:negative)
        end
    end,

    reward = function(s,a)
        if s == :death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        else # :treat
            return 0.1
        end
    end, 

    initialstate = Deterministic(:healthy),
    discount = 0.99
        
)

wait_policy = FunctionPolicy(b -> :wait)
sim = RolloutSimulator(max_steps=100)

estimate = mean(POMDPs.simulate(sim, cancer, wait_policy) for _ in 1:10_000)
println("Estimated value of always-wait policy = ", estimate)

############
# Question 3
############

using Statistics
using CommonRLInterface
using Flux
using Plots
using CommonRLInterface.Wrappers: QuickWrapper
import Cairo, Fontconfig

# Override to a discrete action space, and position and velocity observations rather than the matrix.
env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                   observe=mc -> observe(mc)[1:2]
                  )

# ε-greedy action selection
function select_action(Q, s, ε, env)
    if rand() < ε
        return rand(1:length(actions(env)))   # returns action index
    else
        return argmax(Q(s))                   # best action index
    end
end

# function loss(Q, Q_target, s, a_ind, r, sp, done; γ=0.99)
#     q_sa = Q(s)[a_ind]

#     if done
#         target = r
#     else
#         best_next_a = argmax(Q(sp))              # action chosen by online network
#         target = r + γ * Q_target(sp)[best_next_a]  # value evaluated by target network
#     end

#     return (q_sa - target)^2
# end

# # FIX: added γ as a keyword argument so it exists
function loss(Q, Q_target, s, a_ind, r, sp, done; γ=0.99)
    # Q gained ouput from Nerual network
    q_sa = Q(s)[a_ind]
    # target given general equation with Q_target from deepcopy
    target = done ? r : r + γ * maximum(Q_target(sp))
    # General DQN Loss Function
    return (q_sa - target)^2
end

function dqn(env; num_episodes=3000, n_steps=500, batch_size=32, target_update_freq=10, train_freq =10)
    # Set up Relu nerual network 2 input 1 output
    Q = Chain(
        Dense(2, 64, relu),
        #Dense(32, 32, relu),
        Dense(64, length(actions(env)))
    )

    # Using ADAM for gradient optimization of BNerual netywork
    opt = Flux.setup(ADAM(0.0005), Q)
    Q_target = deepcopy(Q)

    # FIX: keep the replay buffer, do not overwrite it later
    buffer = []

    # set greey policy bounds or RAND
    ε = 1.0
    ε_min = 0.05
    ε_decay = 0.99

    reset!(env)
    s = Float32.(observe(env))

    returns = Float64[]
    episode_return = 0.0

    max_buffer_size = 200_000

    # Iterations for interacting with env
    for episode in 1:num_episodes
        reset!(env)
        s = Float32.(observe(env))
        episode_return = 0.0
        # global_step = 0
        ε = max(ε_min, 1.0 - episode / (num_episodes * 0.8))

        for t in 1:n_steps
            #global_step += 1
            a_ind = select_action(Q, s, ε, env)
            r = act!(env, actions(env)[a_ind])
            sp = Float32.(observe(env))
            done = terminated(env)

            push!(buffer, (s, a_ind, r, sp, done))
            episode_return += r

            # if length(buffer) > max_buffer_size
            #     popfirst!(buffer)
            # end

            # training here
            if length(buffer) >= n_steps*0.2 && t % train_freq == 0
                batch = rand(buffer, batch_size)
                for data in batch
                    loss_value, grads = Flux.withgradient(loss, Q, Q_target, data...)
                    Flux.update!(opt, Q, grads[1])
                end
            end

            

            # update target network
            if t % target_update_freq == 0
                Q_target = deepcopy(Q)
            end

            s = sp

            if done
                break
            end
        end

        push!(returns, episode_return)
    end

    return Q, returns
end

Q, returns = dqn(env)
println("Average return: ", mean(returns))

window = 20

smoothed = [
    mean(returns[max(1, i-window+1):i])
    for i in 1:length(returns)
]

p = plot(returns,
     xlabel="Episode",
     ylabel="Total Reward",
     title="Reward per Episode",
     label="Reward")

display(p)

#DMUStudent.HW5.evaluate(s -> actions(env)[argmax(Q(Float32.(s[1:2])))], n_episodes=100)
#----------
# Rendering
#----------

display(render(env))
using Plots
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
h = heatmap(xs, vs, (x, v) -> maximum(Q(Float32[x, v])),
        xlabel="Position (x)",
        ylabel="Velocity (v)",
        title="Max Q Value")
display(h)

HW5.evaluate(s -> actions(env)[argmax(Q(Float32.(s[1:2])))], "thomas.hoang@colorado.edu")
