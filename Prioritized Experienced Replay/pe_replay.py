"""
优先经验回放
"""

import torch
import numpy as np
import random
from torch import nn
from torch import optim
from torch.nn import functional as F
from collections import namedtuple
import gym
import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

BATCH_SIZE = 32
CAPACITY = 10000
LR = 1e-4
GAMMA = 0.99  # 时间折扣率
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))
torch.manual_seed(123)
ENV = 'CartPole-v0'  # 任务名称
MAX_STEPS = 200  # 1次试验的step
NUM_EPISODES = 500  # 最大试验次数
TD_ERROR_EPSILON = 0.0001  # TD误差偏置


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0,
                        frames[0].shape[0] / 72.0), dpi=72)
    # plt.figure()
    plt.axis('off')

    patch = plt.imshow(frames[0])

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), func=animate, frames=len(frames), interval=50)

    anim.save('movie_cartpole_DDQN.mp4')  # 保存视频
    display(display_animation(anim, default_mode='loop'))


class Net(nn.Module):
    """网络类"""

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_out)
        )

    def forward(self, x):
        output = self.model(x)
        return output


class TDerrorMemory:
    """存储TD误差类"""

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # 最大存储长度
        self.memory = []
        self.index = 0

    def push(self, td_error):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def get_prioritized_indexes(self, batch_size):
        # 根据TD误差以概率获得index

        # 计算TD误差的总和
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 添加一个微小值

        # 生成batch_size的随机数，并按升序排列
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        # 通过得到的随机数获取index
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                    abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

            if idx >= len(self.memory):
                idx = len(self.memory) - 1

            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors


class ReplayMemory:
    """经验池类"""

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # 最大容量
        self.memory = []  # 存储经验
        self.index = 0  # 索引

    def push(self, state, action, state_next, reward):
        """将transition = (state, action, state_next, reward)保存"""

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 内存未满时添加None

        # 使用namedtuple对象Transition将值和字段名称保存为一对
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 将保存的index移动一位
        # self.index+1 小于self.capacity时就是self.index+1，超过时又从0开始

    # 随机选择transition
    def sample(self, batch_size):
        """随机检索Batch_size大小的样本返回"""
        return random.sample(self.memory, batch_size)

    # 返回当前存储的transition数
    def __len__(self):
        """返回当前memory的长度"""
        return len(self.memory)


class Brain:
    """大脑类"""

    def __init__(self, num_states, num_actions):

        # 向左或向右的两个动作
        self.num_actions = num_actions

        # 创建经验池
        self.memory = ReplayMemory(CAPACITY)

        # 构建神经网络
        n_in, n_mid, n_out = num_states, 32, num_actions

        self.main_q_network = Net(n_in, n_mid, n_out)  # 主Q网络
        self.target_q_network = Net(n_in, n_mid, n_out)  # 目标Q网络

        print(self.main_q_network)  # 输出网络形状

        # 优化器
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=LR)

        # 生成TD误差的存储对象
        self.td_error_memory = TDerrorMemory(CAPACITY)

    def replay(self, episode):
        """经验回放学习网络的连接参数"""
        # 1. 检查经验池的大小
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. 创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch(
            episode)

        # 3. 找到Q(s_t, a_t)作为监督信息
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 更新参数
        self.update_main_q_network()

    def decide_action(self, state, episode):
        """根据当前状态确定动作"""
        # 采用epsilon-贪婪法
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # 将网络切换到推理模式
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:
            # 随机返回0和1
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

    def make_minibatch(self, episode):
        """创建小批量数据"""

        # transitions是list类型，有BATCH_SIZE个Transition类型的变量
        if episode < 30:
            transitions = self.memory.sample(BATCH_SIZE)
        else:
            # 根据TD误差取出小批量数据
            indexes = self.td_error_memory.get_prioritized_indexes(BATCH_SIZE)
            transitions = [self.memory.memory[n] for n in indexes]

        # (state, action, state_next, reward) * BATCH_SZIE ==>
        # (state*BATCH_SZIE, action*BATCH_SZIE, state_next*BATCH_SZIE, reward*BATCH_SZIE)
        # [(1,2,3), (1,2,3), (1,2,3)] ==> [(1,1,1), (2,2,2), (3,3,3)]

        # if BATCH_SIZE == 3
        # transitions: [Transition(state=tensor([s1]), action=tensor([a1]), next_state=tensor([sn1]), reward=tensor([r1])),
        #               Transition(state=tensor([s2]), action=tensor([a2]), next_state=tensor([sn2]), reward=tensor([r2])),
        #               Transition(state=tensor([s3]), action=tensor([a3]), next_state=tensor([sn3]), reward=tensor([r3]))]

        # s1=[x1,x2,x3,x4], a1=[x1], sn1=[x1,x2,x3,x4], r1=[x1]

        # batch: Transition(state=(tensor([s1]), tensor([s2], tensor([s3])),
        #                   action=(tensor([a1]), tensor([a2], tensor([a3])),
        #                   next_state=(tensor([sn1]), tensor([sn2], tensor([sn3])),
        #                   reward=(tensor([r1]), tensor([r2], tensor([r3])))

        batch = Transition(*zip(*transitions))

        # if BATCH_SIZE == 3
        # batch.state: (tensor([s1]), tensor([s2]), tensor([s3])), 其中s1,s2,s3都有四个变量
        # state_batch: tensor([s1, s2, s3])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        """找到Q(s_t, a_t)作为监督信息"""

        # 将网络切换到推理模式
        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # if BATCH_SIZE == 3
        # eg: state_batch = tensor([[ 1,  2,  3,   4],
        #                           [ 5,  6,  7,   8],
        #                           [ 9,  10, 11, 12]])
        #
        # self.main_q_network(self.state_batch) = tensor([[1, 2],
        #                                                 [3, 4],
        #                                                 [5, 6]])
        #
        # self.action_batch = tensor([[1],
        #                             [0],
        #                             [0]])
        #
        # self.main_q_network(self.state_batch).gather(1, self.action_batch) = tensor([[2],
        #                                                                              [3],
        #                                                                              [5]])
        #
        # 将self. action_batch作为索引，从self.main_q_network(self.state_batch)中取出对应索引的值

        # non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        non_final_mask = torch.BoolTensor(
            list(map(lambda s: s is not None, self.batch.next_state)))
        # if self.FLAG:
        #     print(non_final_mask)
        #     self.FLAG = False
        #     import sys
        #     sys.exit(0)

        # batch.next_state是None的位置对应0，不是None的位置对应1
        # eg: batch.next_state = (tensor([[1, 2,  3,  4  ]]),
        #                         tensor([[5, 6,  7,  8  ]]),
        #                         None)
        #     non_final_mask = tensor([1, 1, 0])

        next_state_values = torch.zeros(BATCH_SIZE)

        """从主Q网络中求取下一个状态中最大Q值的动作a_m"""

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 维度从[BATCH_SIZE]变成[BATCH_SIZE * 1], 从一维变成二维
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values  # 满足的话就说明已经训练完美了

    def update_main_q_network(self):
        """更新连接参数"""

        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):

        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_td_error_memory(self):
        # 将网络切换到推理模式
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 创建包含所有经验的批量数据
        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        # 获取网络的输出
        state_action_values = self.main_q_network(
            state_batch).gather(1, action_batch)

        # 创建一个索引掩码以判断是否未完成CartPole并具有next_state
        non_final_mask = torch.BoolTensor(
            tuple(map(lambda s: s is not None, batch.next_state)))

        # 首先全部设为0
        next_state_values = torch.zeros(len(self.memory))

        # 从主Q网络中查找下一个状态中最大Q值的动作a_m，得到a_m的索引
        a_m = torch.zeros(len(self.memory)).type(torch.LongTensor)
        a_m[non_final_mask] = self.main_q_network(
            non_final_next_states).detach().max(1)[1]

        # 取出具有下一个状态的，并将size 32 变成 size 32*1
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 从目标Q网络找到对应a_m索引的Q值
        next_state_values[non_final_mask] = self.target_q_network(
            non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 找到TD误差
        td_errors = (reward_batch + GAMMA * next_state_values) - \
            state_action_values.squeeze()

        # 更新TD误差存储变量
        self.td_error_memory.memory = td_errors.detach().numpy().tolist()


class Agent:
    """智能体类"""

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # 创建大脑

    def update_q_function(self, episode):
        self.brain.replay(episode)

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

    def memorize_td_error(self, td_error):
        self.brain.td_error_memory.push(td_error)  # 存储TD误差

    def update_td_error_memory(self):
        self.brain.update_td_error_memory()  # 更新所有TD误差


class Environment:
    """环境类"""

    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.agent = Agent(num_states, num_actions)  # 创建智能体

    def run(self):
        """执行函数"""
        episode_10_list = np.zeros(10)  # 存储10次实验的连续站立步数，输出平均步数
        complete_episodes = 0  # 连续成功的次数
        episode_final = False  # 是否最后一轮的标志
        frames = []  # 存储图像

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()  # 环境初始化

            state = observation  # 观测值即为状态值
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  # numpy类型转换为tensor类型
            state = torch.unsqueeze(state, 0)  # size 4 变为 size 1x4

            for step in range(MAX_STEPS):

                # if episode_final:
                #     frames.append(self.env.render(mode="rgb_array"))

                action = self.agent.get_action(state, episode)  # 求取动作
                observation_next, _, done, _ = self.env.step(
                    action.item())  # 不使用reward和info，所以用_

                if done:  # 步数超过200，或者倾斜超过某个角度，则done为True
                    state_next = None  # 没有下一个状态，故存储None

                    # 将最近10轮的站立步数添加到列表中
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    if step < 195:
                        reward = torch.FloatTensor(
                            [-1.0])  # 步数小于195步视为失败，奖励为-1
                        complete_episodes = 0  # 清空连续成功的次数
                    else:
                        reward = torch.FloatTensor([1.0])  # 站立超过195步视为成功，奖励为1
                        complete_episodes = complete_episodes + 1  # 增加连续成功的次数
                else:
                    reward = torch.FloatTensor([0.0])  # 通常情况奖励都为0
                    state_next = observation_next
                    state_next = torch.from_numpy(
                        state_next).type(torch.FloatTensor)  # numpy类型转为Tensor类型
                    state_next = torch.unsqueeze(
                        state_next, 0)  # size 4 转为 size 1x4

                # 添加经验
                self.agent.memorize(state, action, state_next, reward)

                # 添加TD误差
                self.agent.memorize_td_error(0)  # 应当存储TD误差，先输入0

                # 更新Q函数
                self.agent.update_q_function(episode)

                # 更新状态
                state = state_next

                # 结束时的操作
                if done and episode_final == False:
                    print("%d Episode: Finished after %d steps: 10次试验的平均step数 = %.1lf" %
                          (episode, step + 1, episode_10_list.mean()))

                    # 更新全部TD误差
                    self.agent.update_td_error_memory()

                    # 每两轮更新一次目标Q网络
                    if episode % 2 == 0:
                        self.agent.update_target_q_function()
                    break

            # 最终试验
            if episode_final:
                # display_frames_as_gif(frames)
                break

            # 连续成功10次，则结束试验
            if complete_episodes >= 10:
                print("10轮连续成功！")
                episode_final = True  # 使下一次试验作为最后一轮，从而绘制动画


# main
cartpole_env = Environment()
cartpole_env.run()
