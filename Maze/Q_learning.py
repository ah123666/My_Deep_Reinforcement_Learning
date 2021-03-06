import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm  # color map


def draw_fig():
    # 声明图的大小以及图的变量名
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # 画出红色的墙壁
    plt.plot([1, 1], [0, 1], color='red', linewidth=2)
    plt.plot([1, 2], [2, 2], color='red', linewidth=2)
    plt.plot([2, 2], [2, 1], color='red', linewidth=2)
    plt.plot([2, 3], [1, 1], color='red', linewidth=2)

    # 画出表示状态的文字s0-s8
    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', ha='center')
    plt.text(2.5, 0.3, 'GOAL', ha='center')

    # 设定画图的范围
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')

    # 当前位置S0用绿色圆圈画出
    line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize='60')
    return fig, line, ax


def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))

    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)  # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素

    return pi


# 实现epsilon-贪婪算法
def get_actions(s, Q, epsilon, pi_0):
    direction = ['up', 'right', 'down', 'left']

    # 确定行动
    if np.random.rand() < epsilon:
        # 以概率epsilon随机移动
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        # 采取Q的最大值对应的动作
        next_direction = direction[np.nanargmax(Q[s, :])]
    # 为动作加上索引
    if next_direction == 'up':
        action = 0
    elif next_direction == 'right':
        action = 1
    elif next_direction == 'down':
        action = 2
    elif next_direction == 'left':
        action = 3

    return action


# 由动作确定下一个状态
def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ['up', 'right', 'down', 'left']
    next_direction = direction[a]

    # 由动作确定下一个状态
    if next_direction == 'up':
        s_next = s - 3
    elif next_direction == 'right':
        s_next = s + 1
    elif next_direction == 'down':
        s_next = s + 3
    elif next_direction == 'left':
        s_next = s - 1

    return s_next


# # 基于Sarsa更新动作价值函数Q
# def Sarsa(s, a, r, s_next, a_next, Q, eta, gamma):
#     if s_next == 8:
#         Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
#     else:
#         Q[s, a] = Q[s, a] + eta * (r + gamma * Q[s_next, a_next] - Q[s, a])
#
#     return Q

# 基于Q学习的动作价值函数Q的更新
def Q_learning(s, a, r, s_next, Q, eta, gamma):
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])

    return Q


# 定义基于Sarsa求解迷宫问题的函数，输出状态，动作的历史记录以及更新后的Q

def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0  # 开始地点
    a = a_next = get_actions(s, Q, epsilon, pi)  # 初始动作
    s_a_history = [[0, np.nan]]  # 记录智能体的移动序列

    while 1:
        a = a_next
        s_a_history[-1][1] = a  # 将动作放在现在的状态下，（最终的index=-1）
        s_next = get_s_next(s, a, Q, epsilon, pi)  # 有效的下一个状态
        s_a_history.append([s_next, np.nan])  # 代入下一个状态，动作未知时为nan

        # 给与奖励，求得下一个动作
        if s_next == 8:
            r = 1  # 到达目标，给与其奖励
            a_next = np.nan
        else:
            r = 0
            a_next = get_actions(s_next, Q, epsilon, pi)

        # 更新价值函数
        # Q = Sarsa(s, a, r, s_next, a_next, Q, eta, gamma)
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)
        # 终止判断
        if s_next == 8:
            break
        else:
            s = s_next

    return [s_a_history, Q]


def main():
    draw_fig()
    plt.show()
    # 行为状态S0-S7， 列为↑，→，↓，←表示移到的方向,1表示可以走，np.nan表示不能走
    theta_0 = np.array([[np.nan, 1, 1, np.nan],  # S0
                        [np.nan, 1, np.nan, 1],  # S1
                        [np.nan, np.nan, 1, 1],  # S2
                        [1, 1, 1, np.nan],  # S3
                        [np.nan, np.nan, 1, 1],  # S4
                        [1, np.nan, np.nan, np.nan],  # S5
                        [1, np.nan, np.nan, np.nan],  # S6
                        [1, 1, np.nan, np.nan]  # S7
                        ])

    # 设置初始的动作价值函数
    [a, b] = theta_0.shape
    # 初始Q值
    Q = np.random.rand(a, b) * theta_0 * 0.1  # 将theta_0乘到各元素上，使得Q的墙壁方向的值为nan

    # 求解随机动作策略pi_0
    pi_0 = simple_convert_into_pi_from_theta(theta_0)

    # 通过Q学习求解迷宫问题
    eta = 0.1  # 学习率
    gamma = 0.9  # 时间折扣率
    epsilon = 0.5  # epsilon-贪婪法的初始值
    v = np.nanmax(Q, axis=1)  # 求每个状态价值的最大值
    is_continue = True
    episode = 1

    V = [np.nanmax(Q, axis=1)]  # 存放每回合的状态价值
    while is_continue:
        print('当前回合:', episode)
        # epsilon值逐渐减少
        epsilon = epsilon / 2
        # 求取移动历史和更新后的Q值
        [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)
        # 状态价值的变化
        new_v = np.nanmax(Q, axis=1)
        print('状态价值的变化:', np.sum(np.abs(new_v - v)))
        print('步数为:', len(s_a_history) - 1, '\n')
        v = new_v
        V.append(v)

        # 重复100回合
        episode = episode + 1
        if episode > 100:
            break
    print('动作轨迹为:', s_a_history)
    print('动作价值Q:\n', Q)
    print('状态价值V:\n', v)

    fig, line, ax = draw_fig()

    def init():
        # 初始化背景图像
        line.set_data([], [])
        return line

    def animate(i):
        # 各帧的绘图内容
        # 各方格中根据状态价值的大小画颜色
        line, = ax.plot([0.5], [2.5], marker="s",
                        color=cm.jet(V[i][0]), markersize=85)  # S0
        line, = ax.plot([1.5], [2.5], marker="s",
                        color=cm.jet(V[i][1]), markersize=85)  # S1
        line, = ax.plot([2.5], [2.5], marker="s",
                        color=cm.jet(V[i][2]), markersize=85)  # S2
        line, = ax.plot([0.5], [1.5], marker="s",
                        color=cm.jet(V[i][3]), markersize=85)  # S3
        line, = ax.plot([1.5], [1.5], marker="s",
                        color=cm.jet(V[i][4]), markersize=85)  # S4
        line, = ax.plot([2.5], [1.5], marker="s",
                        color=cm.jet(V[i][5]), markersize=85)  # S5
        line, = ax.plot([0.5], [0.5], marker="s",
                        color=cm.jet(V[i][6]), markersize=85)  # S6
        line, = ax.plot([1.5], [0.5], marker="s",
                        color=cm.jet(V[i][7]), markersize=85)  # S7
        line, = ax.plot([2.5], [0.5], marker="s",
                        color=cm.jet(1.0), markersize=85)  # S8
        return line

    anim = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(V), interval=200, repeat=False)
    plt.show()


if __name__ == '__main__':
    main()




