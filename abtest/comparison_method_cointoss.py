# this program is coped from
# https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/epsilon_greedy.py
# this is a quizz about epsilon-greedy
# there are several coins which have different probability to show head or tail
# the player explores or exploits those coins and pushes the game to an advantage

# add UCB1 (Upper Confidence Bound version.1) based on the following site
# https://docs.microsoft.com/ja-jp/archive/msdn-magazine/2019/august/test-run-the-ucb1-algorithm-for-multi-armed-bandit-problems
# this is just for comparison to the Epsilon-Greedy proram

# add beta distribution method for comparison
# the code is almost copied from Udemy course, Bayesian A/B Testing
# this is bayesian a/b testing method cause beta distribution is
# conjugate prior for Bernoui trial

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

# beta distribution method
class BayesAgent():
    def __init__( self, env ):
        # first row is a (num of head) and second is b (nom of tail)
        self.ab = np.ones( ( len(env), 2 ) )
    
    def policy( self ):
        bestb = None
        maxsample = -1
        for i in range( self.ab.shape[0] ):
            sample = np.random.beta( self.ab[i][0], self.ab[i][1] )
            if sample > maxsample:
                maxsample = sample
                bestb     = i
        return bestb
    
    def play( self, env ):
        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step( selected_coin )
            rewards.append( reward )

            self.ab[selected_coin][0] += reward
            self.ab[selected_coin][1] += 1 - reward

        return rewards

# this class choose the best bandit continuously for comparison with respect to regret
class BestAgent():
    def __init__(self, env):
        self.bestb = np.argmax( env.head_probs )
        self.env = env

    def play(self):
        self.env.reset()
        done = False
        rewards = []
        while not done:
            reward, done = self.env.step( self.bestb )
            rewards.append( reward )
        return rewards



if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    def main():
        #env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        env = CoinToss( [0.1, 0.3, 0.25, 0.4, 0.5, 0.35, 0.45, 0.6, 0.75] )
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

        # beta distribution method (bayesian method)
        agent = BayesAgent( env )
        means = []
        for s in game_steps:
            env.max_episode_steps = s
            rewards = agent.play( env )
            means.append( np.mean( rewards ) )
        result["beta dist"] = means

        # maximum benefit if the best bandit was countinuously choosed
        agent = BestAgent( env )
        means = []
        for s in game_steps:
            env.max_episode_steps = s
            rewards = agent.play()
            means.append( np.mean( rewards ) )
        result['best choise'] = means

        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        plt.show()

    main()