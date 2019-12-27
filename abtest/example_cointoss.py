# this program is coped from
# https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/epsilon_greedy.py
# this is a quizz about epsilon-greedy
# there are several coins which have different probability to show head or tail
# the player explores or exploits those coins and pushes the game to an advantage

# add UCB1 (Upper Confidence Bound version.1) based on the following site
# https://docs.microsoft.com/ja-jp/archive/msdn-magazine/2019/august/test-run-the-ucb1-algorithm-for-multi-armed-bandit-problems
# this is just for comparison to the Epsilon-Greedy proram

import random
import numpy as np


class CoinToss():

    def __init__(self, head_probs, max_episode_steps=30):
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("The step count exceeded maximum. \
                            Please reset env.")
        else:
            done = True if self.toss_count == final else False

        if action >= len(self.head_probs):
            raise Exception("The No.{} coin doesn't exist.".format(action))
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done


class EpsilonGreedyAgent():

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = []

    def policy(self):
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(coins)
        else:
            return np.argmax(self.V)

    def play(self, env):
        # Initialize estimation.
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average

        return rewards

# UCB1 class
class UCB1Agent():
    def __init__( self, env ):
        self.V = len( env )
        self.N = len( env )

    def policy( self, trial ):
        if trial <= len( self.N ):
            idx = np.argmin( self.N )
            return idx

        # calucurate dicision values for all arms or coins
        decValue = [0] * len( self.N )
        # the first (num of arms) trials are counted as initial setup
        # which means the first (num of arms) trials are not counted in
        # following calucration
        t = trial - len( self.N )
        for i in range( len( self.N ) ):
            decValue[i] = self.V[i] + np.sqrt( 2 * np.log( t ) / self.N[i] )

        return np.argmax( decValue )

    def play( self, env ):
        self.N = [0] * len( env )
        self.V = [0] * len( env )

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy( env.toss_count + 1 )
            reward, done = env.step( selected_coin )
            rewards.append( reward )

            n = self.N[ selected_coin ]
            coin_average = self.V[ selected_coin ]
            new_average  = ( coin_average * n + reward ) / ( n + 1 )
            self.N[ selected_coin ] += 1
            self.V[ selected_coin ]  = new_average
        
        return rewards


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    def main():
        env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
        game_steps = list(range(10, 310, 10))
        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                rewards = agent.play(env)
                means.append(np.mean(rewards))
            result["epsilon={}".format(e)] = means
        
        # UCB1 algorithm
        agent = UCB1Agent( env )
        means = []
        for s in game_steps:
            env.max_episode_steps = s
            rewards = agent.play( env )
            means.append( np.mean( rewards ) )
        #result["epsilon={}".format(-1)] = means
        result["UCB1"] = means

        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        plt.show()

    main()