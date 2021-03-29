# 包导入
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import gym
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
# 声明动画的绘图函数
# 参考URL http://nbviewer.jupyter.org/github/patrickmineault
# /xcorr-notebooks/blob/master/Render%20OpenAI%20gym%20as%20GIF.ipynb
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from collections import namedtuple


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    anim.save('movie_cartpole_DQN.mp4')  # 保存视频
    display(display_animation(anim, default_mode='loop'))
    # plt.show()
    # plt.close()


# 生成namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# 变量的设定
ENV = 'CartPole-v0'  # 任务名称
GAMMA = 0.99  # 时间折扣率
MAX_STEPS = 200  # 1次试验的step
NUM_EPISODES = 500  # 最大试验次数



# 定义用于存储经验的内存类
class ReplayMemory:

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


BATCH_SIZE = 3
CAPACITY = 10000



# 智能体大脑类，执行DQN
class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 获取CartPole的两个动作（向左或向右）

        # 创建存储经验的对象
        self.memory = ReplayMemory(CAPACITY)

        # 构建一个神经网络
        # self.model = nn.Sequential()
        # self.model.add_module('fc1', nn.Linear(num_states, 32))
        # self.model.add_module('relu1', nn.ReLU())
        # self.model.add_module('fc2', nn.Linear(32, 32))
        # self.model.add_module('relu2', nn.ReLU())
        # self.model.add_module('fc3', nn.Linear(32, num_actions))

        self.model = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

        # print(self.model)  # 输出网络的形状

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        """通过Experience Replay学习网络的连接参数"""

        # -----------------------------------------
        # 1. 检查经验池的大小
        # -----------------------------------------
        # 1.1 经验池大小小于小批量数据时不执行任何操作
        if len(self.memory) < BATCH_SIZE:
            return

        # -----------------------------------------
        # 2. 创建小批量数据
        # -----------------------------------------
        # 2.1 从经验池中获取小批量数据
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 将每个变量转换为与小批量数据对应的形式
        # 得到的transitions为一个BATCH_SIZE的(state, action, state_next, reward)
        # (state, action, state_next, reward) × BATCH_SIZE
        # ==> (state × BATCH_SIZE, action × BATCH_SIZE, state_next × BATCH_SIZE, reward × BATCH_SIZE)
        batch = Transition(*zip(*transitions)) # *zip(*transitions) is zip type

        # print("transitions:\n", transitions)  # transitions is list type
        # print("\n")
        # print(transitions.type())
        # print("\n"*3)
        # print("batch:\n", batch)  # batch is Transition type
        # print("batch:\n", *zip(*transitions))  # zip type
        # print("\n")
        # print(batch.type()) 
        # print("\n"*3)
        # 2.3 将每个变量的元素转换为小批量数据对应的形式
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])


        # -----------------------------------------
        # 3. 求取Q(s_t, a_t)的值作为监督信号
        # -----------------------------------------
        # 3.1 将网络切换到推理模式
        self.model.eval()

        # 3.2 求取网络的输出Q(s_t, a_t)
        # state_action_values = self.model(state_batch).gather(1, action_batch)
        state_batch_after_net = self.model(state_batch)
        state_action_values = state_batch_after_net.gather(1, action_batch)

        # print("BATCH_SIZE:\n", BATCH_SIZE)
        # print("\n"*3)
        # print("state_batch:\n", state_batch)
        # print("\n"*3)
        # print("state_batch.type():\n", state_batch.type())
        # print("\n"*3)
        # print("action_batch:\n", action_batch)
        # print("\n"*3)
        # print("action_batch.type():\n", action_batch.type())
        # print("\n"*3)
        # print("state_batch_after_net:\n", state_batch_after_net)
        # print("\n"*3)
        # print("state_batch_after_net.type()\n", state_batch_after_net.type())
        # print("\n"*3)



        # 3.3 求取max{Q(s_t+1, a)}的值

        # 创建索引掩码以检查cartpole是否完成且具有next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        # print("batch.next_state:\n", batch.next_state)
        # print("\n"*3)
        # print("non_final_mask:\n", non_final_mask)
        import sys
        sys.exit(0)
        # 首先全部设置为0
        next_state_values = torch.zeros(BATCH_SIZE)

        # 求取具有下一状态的index的最大值
        # 访问输出并通过max()求列方向最大值[Value, index]
        # 并输出其Q值(index = 0)
        # 用detach取出该值
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        # 3.4 从Q公式中求取Q(s_t, a_t)值作为监督信息
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # -----------------------------------------
        # 4. 更新连接参数
        # -----------------------------------------
        # 4.1 将网络切换到训练模式
        self.model.train()

        # 4.2 计算损失函数
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # 4.3 更新连接参数
        self.optimizer.zero_grad()
        loss.backward()  # 反向传播
        self.optimizer.step()

    def decide_action(self, state, episode):
        """根据当前状态确定动作"""
        # 采用ε-贪婪法逐步采用最佳动作
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # 将网络切换到推理模式
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # 获取网络输出最大值的索引index = max(1)[1]

        else:
            # 随机返回0,1动作
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action


# 智能体类，一个带有杆的小车
class Agent:
    def __init__(self, num_states, num_actions):
        """设置任务状态和动作的数量"""
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        """更新Q函数"""
        self.brain.replay()

    def get_action(self, state, episode):
        """确定动作"""
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        """将state, action, state_next, reward的内容保存到经验池中"""
        self.brain.memory.push(state, action, state_next, reward)


# 执行CartPole的环境类
class Environment:

    def __init__(self):
        self.env = gym.make(ENV)  # 设定要执行的任务
        num_states = self.env.observation_space.shape[0]  # 任务状态
        num_actions = self.env.action_space.n  # 动作数量为2
        # print(num_states, num_actions)
        self.agent = Agent(num_states, num_actions)

    def run(self):
        """执行"""
        episode_10_list = np.zeros(10)  # 存储10次试验的连续站立步数，用于输出平均步数
        complete_episodes = 0  # 连续成功完成的次数
        episode_final = False  # 最终试验的标志
        frames = []  # 用于存储视频图像的变量

        for episode in range(NUM_EPISODES):  # 试验的最大重复次数
            observation = self.env.reset()  # 环境初始化

            state = observation  # 直接使用观测作为状态state的使用
            state = torch.from_numpy(state).type(torch.FloatTensor)  # 将numpy变量转换为Pytorch Tensor
            state = torch.unsqueeze(state, 0)  # size 4转换为size 1x4

            for step in range(MAX_STEPS):  # 每个回合的循环

                if episode_final is True:  # 将最终试验各个时刻的图像添加到帧中
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(state, episode)  # 求取动作

                # 通过执行动作a_t找到s_(t+1)和done标志
                observation_next, _, done, _ = self.env.step(action.item())  # 不使用regain和info

                # 给与奖励，对episode是否结束以及是否有下一个状态进行判断
                if done:  # 如果步数超过200或者倾斜角度超过某个角度，则done为True
                    state_next = None  # 没有下一个状态，因此存储none

                    # 添加到最近的10轮的站立步数列表中
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    if step < 195:
                        reward = torch.FloatTensor([-1.0])  # 如果半途倒下，奖励为-1
                        complete_episodes = 0  # 清空连续完成的次数
                    else:
                        reward = torch.FloatTensor([1.0])  # 一直站立知道结束时奖励为1
                        complete_episodes = complete_episodes + 1  # 增加连续完成次数的记录
                else:
                    reward = torch.FloatTensor([0.0])  # 普通奖励为0
                    state_next = observation_next  # 保持观察不变
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                # 向经验池中添加经验
                self.agent.memorize(state, action, state_next, reward)

                # Experience Replay中更新Q函数
                self.agent.update_q_function()

                # 更新观测值
                state = state_next

                # 结束处理
                if done and episode_final == False:
                    print('%d Episode: Finished after %d steps：10次试验的平均step数 = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))
                    break

            if episode_final is True:
                # 保存并绘制动画
                display_frames_as_gif(frames)
                break

            # 连续10轮成功
            if complete_episodes >= 10:
                print('10轮连续成功！')
                episode_final = True



# main
cartpole_env = Environment()
cartpole_env.run()
