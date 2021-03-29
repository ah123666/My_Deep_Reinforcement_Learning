import torch
import numpy as np

# # 假设BATCH_SIZE==3
# state_batch = torch.Tensor([[1, 2, 3, 4],
#                             [5, 6, 7, 8],
#                             [9, 10, 11, 12]])
# # 经过网络后
# state_batch_after_net = torch.Tensor([[1, 2],
#                               [3, 4],
#                               [5, 6]])

# action_batch = torch.LongTensor([[1],
#                                  [0], 
#                                  [0]])

# print(state_batch)

# print(state_batch_after_net.gather(1, action_batch))

next_state = (torch.Tensor([[1, 2,  3,  4  ]]),
              torch.Tensor([[5, 6,  7,  8  ]]),
              # torch.Tensor([[9, 10, 11, 12]]),
              None)
print(next_state)

non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, next_state)))

print(non_final_mask)
print(torch.zeros(1,3))