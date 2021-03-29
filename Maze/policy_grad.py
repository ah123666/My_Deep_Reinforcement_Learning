import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


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
    circle, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)
    return fig, circle


# 根据Softmax函数将theta转换为pi
def softmax_convert_into_pi_from_theta(theta):

    beta = 1.0
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    exp_theta = np.exp(beta * theta)

    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
        # 如何在求和、均值时忽略nan？
        # 使用：np.nansum()、np.nanmean()
    pi = np.nan_to_num(pi)

    return pi


# 定义求取动作a以及1步移动后的状态s的函数
def get_action_and_next_s(pi, s):

    direction = ["up", "right", "down", "left"]
    # 根据概率pi[s, :]选择direction
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        action = 0
        s_next = s - 3
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "down":
        action = 2
        s_next = s + 3
    # elif next_direction == "left":
    else:
        action = 3
        s_next = s - 1
    return [action, s_next]


# 定义求解迷宫问题的函数，它输出状态和动作
def goal_maze_ret_s_a(pi):
    s = 0  # 开始地点
    s_a_history = [[0, np.nan]]
    while 1:
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action  # 代入当前状态的动作
        s_a_history.append([next_s, np.nan])  # 代入下一个状态，由于不知道其动作，故用nan表示

        if next_s == 8:
            break
        else:
            s = next_s

    return s_a_history


def update_theta(theta, pi, s_a_history):

    eta = 0.1  # 学习率
    T = len(s_a_history) - 1  # 到达目标的总步数

    [m, n] = theta.shape
    delta_theta = theta.copy()

    # 求取delta_theta的各元素
    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i, j])):  # theta不是nan时
                SA_i = [SA for SA in s_a_history if SA[0] == i]
                # 从列表中取出状态i
                SA_ij = [SA for SA in s_a_history if SA == [i, j]]
                # 取出状态i下应该采取的动作j

                N_i = len(SA_i)  # 状态i下动作的总次数
                N_ij = len(SA_ij)  # 状态i下采取动作j的总次数
                delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T

    new_theta = theta + eta * delta_theta

    return new_theta


def main():
    draw_fig()
    plt.show()
    # 行为状态S0-S7， 列为↑，→，↓，←表示移到的方向,1表示可以走，np.nan表示不能走
    #                    up      right   down    left
    theta_0 = np.array([[np.nan, 1,      1,      np.nan],        # S0
                        [np.nan, 1,      np.nan, 1],             # S1
                        [np.nan, np.nan, 1,      1],             # S2
                        [1,      1,      1,      np.nan],        # S3
                        [np.nan, np.nan, 1,      1],             # S4
                        [1,      np.nan, np.nan, np.nan],        # S5
                        [1,      np.nan, np.nan, np.nan],        # S6
                        [1,      1,      np.nan, np.nan]         # S7
                        ])
    pi_0 = softmax_convert_into_pi_from_theta(theta_0)

    theta = theta_0
    pi = pi_0

    stop_epsilon = 10**-4
    is_continue = True
    count = 0
    while is_continue:
        s_a_history = goal_maze_ret_s_a(pi)
        new_theta = update_theta(theta, pi, s_a_history)
        new_pi = softmax_convert_into_pi_from_theta(new_theta)

        print('策略的变化为：', np.sum(np.abs(new_pi - pi)))
        print('花费的步数为：', len(s_a_history) - 1)

        if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
            is_continue = False
        else:
            theta = new_theta
            pi = new_pi
        count += 1

    np.set_printoptions(precision=3, suppress=True)  # 设置有效位数为3，不显示指数
    print('最终策略为：\n', pi)
    print('最终步数为：\n', len(s_a_history) - 1)
    print('智能体移动记录：\n', s_a_history)
    print('迭代次数：\n', count)

    fig, circle = draw_fig()

    def init():
        # 初始化背景图像
        circle.set_data([], [])
        return circle

    def update(n):
        state = s_a_history[n][0]
        x = (state % 3) + 0.5
        y = 2.5 - int(state / 3)
        circle.set_data(x, y)
        return circle

    anim = animation.FuncAnimation(fig=fig, func=update, init_func=init,
                                   frames=len(s_a_history), interval=300, repeat=True)
    # 这里的frames在调用update函数时会将frames作为实参传递给“n”
    plt.show()


if __name__ == '__main__':
    main()







