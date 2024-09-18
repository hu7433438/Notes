# import numpy as np
#
# # Define the grid size and parameters
# grid_size = 5  # Adjust as needed
# gamma = 0.5
# p_stay = 0.5
# p_stay_end = 0.5
# p_move = 1/3
#
# # Initialize the value function
# V = np.array([0,0,0,0,1])
#
# # Value iteration loop
# for _ in range(100):
#     new_V = np.zeros(grid_size)
#     for state in range(grid_size):
#         # actions = ["stay", "left", "right"]
#         actions = ["stay", "move"]
#         max_value = -np.inf
#         for action in actions:
#             value = V[state]
#             if action == "stay":
#                 if state == 0:
#                     next_state_prob = [0, p_stay, (1 - p_stay) * p_stay_end]
#                 if state == grid_size - 1:
#                     next_state_prob = [(1 - p_stay) * p_stay_end, p_stay, 0]
#                 else:
#                     next_state_prob = [p_stay * p_stay_end / 2, p_stay, p_stay * p_stay_end / 2]
#             elif action == "move":
#                 next_state_prob = [p_stay, p_stay / 2, p_stay / 2] if state == 0 else [p_move / 2, p_stay, p_move / 2]
#             else:
#                 next_state_prob = [p_stay / 2, p_stay, p_stay / 2] if state == grid_size - 1 else [p_move / 2, p_move / 2, p_stay]
#             for next_state, prob in enumerate(next_state_prob):
#                 value += gamma * prob * V[next_state]
#             max_value = max(max_value, value)
#         new_V[state] = max_value
#     V = new_V
#
# # Print the final value function
# print(V)

# import numpy as np
# state = [0, 1, 2, 3, 4]
# action = [0, 1, 2]
# # representing moving left, staying, moving right respectively
# # transition probability
# T = np.array([[[1/2,1/2,0,0,0], [1/2,1/2,0,0,0], [2/3,1/3,0,0,0]],
#                [[1/3,2/3 ,0,0,0], [1/4,1/2,1/4,0,0], [0,2/3,1/3,0,0]],
#                [[0,1/3,2/3,0,0], [0,1/4,1/2,1/4,0], [0,0,2/3,1/3,0]],
#                [[0,0,1/3,2/3,0], [0,0,1/4,1/2,1/4], [0,0,0,2/3,1/3]],
#                [[0,0,0,1/3,2/3], [0,0,0,1/2,1/2], [0,0,0,1/2,1/2]]])
# num_state = 5
# num_action = 3
# r = 1/2
# # initialization
# V = np.zeros(5)
# # reward
# R = np.zeros(5)
# R[4] = 1
# num_iter = 100
# for i in range(num_iter):
#     Q = [[sum([T[s][a][t] * (R[s] + r * V[t]) for t in range(num_state)]) for a in range(num_action)] for s in range(num_state)]
#     V = np.max(Q, axis=1)
# print(V)
import numpy as np

def y(sentence):
    s = np.zeros(2)
    w_s = np.array([[-1,0],[0,1]])
    w_x = np.array([[1, 0], [0, 1]])
    for x in sentence:
        s = np.maximum(w_s@s + w_x@x, 0)
        print(s)
    print("final", s)

y(np.array([[1,0],[1,0]]))
y(np.array([[1,0],[0,1],[0,1]]))
y(np.array([[0,1],[1,0],[1,0]]))

jQuery.ajax(
        {
                type:'POST',
                dataType:'text',
                url:'//store.steampowered.com/checkout/addfreelicense',
                data:
                        {
                                action:'add_to_cart',
                                sessionid:g_sessionID,
                                subid:49307
                }
        }
)

