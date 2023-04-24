from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt

MEAN = 0
VARIANCE = 1
K = 10 # Number of Arms

# Settings for plot customization
plt.style.use('fivethirtyeight') # My favorite plot theme <3
LINE_WIDTH = 2 # How thick each lines are


# Returns a solution as a tuple of size 2
#   0 - average_observed reward
#   1 - optimal_selection_ratio
def epsilon_greedy(epsilon = 0, num_instances = 2000, time_horizon = 1000):
    # INITIALIZE VARIABLES FOR EACH SIMULATION RUN HERE:

    # y-axis for a
    average_observed_reward = [0 for i in range(time_horizon)]

    # y-axis for b
    optimal_selection_ratio = [0 for i in range(time_horizon)]
    
    for instance in range(num_instances):
        # INITIALIZE VARIABLES FOR EACH INSTANCE HERE:
        
        # q*(a) - Action Values
        # Mean: 0
        # Variance = 1
        # Num Entries = 10
        q_star = list(random.normal(MEAN, VARIANCE, K))
        A = [] # Action Taken
        R = [] # Reward
        Q = [0 for i in range(K)] # Estimated Value of Action
        k_occur = [0 for i in range(K)] # Number of times a specific arm was chosen
        optimal_arm = q_star.index(max(q_star)) # Index of the optimal arm to choose
        num_optimal = 0 # Number of times the optimal choice was chosen
        total_reward = 0

        # Plot values for graphs
        average_R = []
        optimal_proportion = []
        
        for i in range(time_horizon):
            arm_index = -1
            
            # Decide whether to Explore or Exploit
            
            # Case 1: Exploit
            if i > 0 and random.normal(0.5, 1) % 1 <= epsilon:
                # print("\tEXPLOIT")
                arm_index = Q.index(max(Q)) # We take the current best arm in Q

            # Case 2: Explore
            else:
                # print("EXPLORE")
                arm_index = random.randint(0, K) # Randomly selects an arm from 1 - K
            
            # Takes a sample from the arm
            curr_reward = random.normal(q_star[arm_index], 1)

            # Increment occurance of this arm
            k_occur[arm_index] += 1

            # Updates Estimated Value of Action of arm
            Q[arm_index] += (curr_reward - Q[arm_index]) / k_occur[arm_index]

            # Records Action Taken
            A.append(arm_index)

            # Records Reward
            R.append(curr_reward)

            # Increment num_optimal if the actual best arm was chosen
            if arm_index == optimal_arm:
                num_optimal += 1
            
            # Updates total reward
            total_reward += curr_reward

            # Updates plot values
            average_R.append(total_reward / (i + 1))
            optimal_proportion.append(num_optimal / (i + 1))

            # Updates simulation plot variables
            average_observed_reward[i] += average_R[i] / num_instances
            optimal_selection_ratio[i] += optimal_proportion[i] / num_instances

    return (average_observed_reward, optimal_selection_ratio)