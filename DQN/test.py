from collections import namedtuple
from torch import nn
Tr = namedtuple('wew', ('name_a', 'value_b'))
Tr_object = Tr('A', 100)
print(Tr_object)
print(Tr_object.name_a)
print(Tr_object.value_b)
# num_states = 16
# num_actions = 2
# model = nn.Sequential()
# model.add_module('fc1', nn.Linear(num_states, 32))
# model.add_module('relu1', nn.ReLU())
# model.add_module('fc2', nn.Linear(32, 32))
# model.add_module('relu2', nn.ReLU())
# model.add_module('fc3', nn.Linear(32, num_actions))

# model1 = nn.Sequential(nn.Linear(num_states, 32),
#                        nn.ReLU(),
#                        nn.Linear(32, 32),
#                        nn.ReLU(),
#                        nn.Linear(32, num_actions))
# print(model)
# print(model1)
