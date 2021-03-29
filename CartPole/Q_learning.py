
import matplotlib.pyplot as plt
import gym
import numpy as np

# 声明动画的绘图函数

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    # plt.figure(figsize=(frames[0].shape[1], frames[0].shape[0]))
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    anim.save('movie_cartpole.mp4')  # 保存视频
    display(display_animation(anim, default_mode='loop'))
    # plt.show()


# 常量的设定
ENV = 'CartPole-v0'  # 任务名称
NUM_DIZITIZED = 6  # 将每个状态划分为离散值的个数
GAMMA = 0.99  # 时间折扣率
ETA = 0.5  # 学习系数
MAX_STEPS = 200  # 1次试验的step
NUM_EPISODES = 1000  # 最大试验次数


# 智能体类，一个带有杆的小车
class Agent:

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # 大脑

    def update_Q_function(self, observation, action, reward, observation_next):
        """更新Q函数"""
        self.brain.update_Q_table(
            observation, action, reward, observation_next)

    def get_action(self, observation, step):
        """确定动作"""
        action = self.brain.decide_action(observation, step)
        return action


# 智能体大脑类，进行Q学习
class Brain:

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 获取CartPole的两个动作（向左或向右）

        # 创建Q表，行数是将状态转换为数字得到的分割数（有4个变量），列数表示动作数
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIZITIZED ** num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        """求得观察到的状态（连续值）到离散值的数字转换阈值"""
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        """将观察到的observation转换为离散值"""
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIZITIZED)),  # 小车位置
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIZITIZED)),  # 小车速度
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZED)),  # 杆的角度
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIZITIZED))  # 杆的角速度
        ]
        return sum([x * (NUM_DIZITIZED ** i) for i, x in enumerate(digitized)])
        # 将小车的四个状态变量放在一起转换到0~1295之间，每状态变量有6个离散值，共有4个状态变量，共6**4=1296种状态

    def update_Q_table(self, observation, action, reward, observation_next):
        """Q学习更新Q表"""
        state = self.digitize_state(observation)  # 状态离散化
        state_next = self.digitize_state(observation_next)  # 将下一状态离散化
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
                                      ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        """根据ε-贪婪法逐渐采取最佳动作"""
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)  # 随机返回0，1动作
        return action


# 执行CartPole的环境类
class Environment:

    def __init__(self):
        self.env = gym.make(ENV)  # 设置要执行的任务
        num_states = self.env.observation_space.shape[0]  # 获取任务状态的个数
        num_actions = self.env.action_space.n  # 获取CartPole的动作数为2（向左或向右）
        self.agent = Agent(num_states, num_actions)  # 创建智能体

    def run(self):
        """执行"""
        complete_episodes = 0
        is_episode_final = False  # 是否为最终试验的标志
        frames = []  # 用于存储视频图像的变量

        for episode in range(NUM_EPISODES):  # 试验的最大重复次数
            observation = self.env.reset()  # 环境初始化

            for step in range(MAX_STEPS):  # 每个回合的循环

                if is_episode_final is True:  # 将最终实验各个时刻的图像添加到帧中
                    frames.append(self.env.render(mode='rgb_array'))

                # 求取动作
                action = self.agent.get_action(observation, episode)

                # 通过执行动作a_t 找到 s_{t+1}, r_{t+1}
                observation_next, _, done, _ = self.env.step(action)  # 不使用reward和info

                # 给予奖励
                if done:  # 如果步数超过200，或者如果倾斜超过某个角度，则done为True
                    if step < 195:
                        reward = -1  # 如果半途摔倒，给予奖励-1作为惩罚
                        complete_episodes = 0  # 清空连续记录
                    else:
                        reward = 1  # 站立超过195步即为成功，给予奖励1
                        complete_episodes += 1  # 增加连续记录
                else:
                    reward = 0  # 途中奖励为0

                # 使用step+1的状态observation_next更新Q函数
                self.agent.update_Q_function(observation, action, reward, observation_next)

                # 更新observation
                observation = observation_next

                # 结束时的操作
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
                    break

            if is_episode_final is True:  # 播放并保存动画
                display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:  # 连续成功10次
                print('连续成功10次！')
                is_episode_final = True  # 结束试验


cartpole_env = Environment()
cartpole_env.run()
