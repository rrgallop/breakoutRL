import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
import random
np.set_printoptions(threshold=np.nan)


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()

    """Initialize neural network using keras"""
    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', init='he_uniform'))
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(16, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    """Store batches of information for backpropagation"""
    def remember_state(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        # Create fake label
        y[action] = 1
        # Derive gradient & save
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    """Action probabilities produced by network"""
    def choose_action(self, state, prob=None):
        state = state.reshape([1, state.shape[0]])
        if prob is not None:
            aprob = prob
        else: aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        # Ensure the probabilities add to 1
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, aprob

    """Used for epsilon greedy"""
    def random_act(self, rand_prob):
        aprob = rand_prob
        self.probs.append(aprob)
        action = np.random.choice(self.action_size, 1, p=aprob)[0]
        return action, aprob

    """Working backwards from a rewarding frame, frames leading up to the rewarding
       frame are assigned some reward."""
    def discount_rewards(self, rewards):
        normal = False
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        if normal:
            mean = np.mean(discounted_rewards)
            std = np.std(discounted_rewards)
            discounted_rewards = (discounted_rewards - mean) / (std)
        return discounted_rewards


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



"""Preprocess image by cropping, downsampling and converting to grayscale"""
def preprocess(image):
    image = image[35:195]
    image = image[::2, ::2, 0]
    image[image == 144] = 0
    image[image == 109] = 0
    image[image != 0] = 1
    return image.astype(np.float).ravel()

"""Used for epsilon greedy"""
def initialize_random_probability(action_size):
    action_probability =  1.0/action_size
    random_chance = [action_probability] *action_size
    return random_chance


if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    state = env.reset()
    prev_x = None
    score = 0
    episode = 0
    frames = 0

    state_size = 80 * 80
    action_size = env.action_space.n

    # initialize random_chance as the list of actions with equal probabilities
    random_chance = initialize_random_probability(action_size)
    # initialize agent
    agent = PGAgent(state_size, action_size)

    # used to track the raw reward value assigned by OpenAI Gym
    real_reward = 0

    # lives is updated every iteration by OpenAI Gym, curr_lives is how many lives we think we have
    curr_lives = 5
    lives = 5

    # Behavior flags
    visualize = False
    frame_reward = True
    normalize = False
    random_games = False
    one_life = True
    death_penalty = False

    """Uncomment to continue your session"""
    # agent.load('forced shoot reward structure 1.h5')

    # Used to introduce random decision making to the agent. We will generate random numbers between 1 and eps_initial.
    # If the random number is less than epsilon's current value, the agent will take a random action.
    epsilon = 0
    eps_max = 10000
    eps_min = 100

    penalize = False
    avg_reward = 0
    running_reward = 0

    # save output to file
    f = open("game-results.txt", "a")

    while True:
        if visualize: env.render()
        frames = frames + 1
        cur_x = preprocess(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x
        if random_games:
            random_number = random.randint(1,eps_max)
            if (random_number < epsilon):
                action, prob = agent.random_act(random_chance)
            else:
                action, prob = agent.choose_action(x)
        else:
            if death_penalty:
                if lives < curr_lives:
                    action = 1
                    curr_lives = lives
                    prob = [0, 1, 0, 0]
                    action, prob = agent.choose_action(x, prob)
                    penalize = True
                else: action, prob = agent.choose_action(x)
            else: action, prob = agent.choose_action(x)



        state, reward, done, info = env.step(action)
        if penalize:
            reward = reward - 1
            penalize = False
        lives = info['ale.lives']

        # Option to end game after a single lost life
        if one_life == True and lives < curr_lives:
            done = True

        real_reward += reward
        if frame_reward:
            frames_reward =+ frames *.01
            if (reward > 0): reward += frames_reward
            if reward >= 1: print(reward)
        score += reward
        agent.remember_state(x, action, prob, reward)
        running_reward = running_reward + reward

        if done:
            avg_reward = running_reward / episode if episode > 0 else 0
            curr_lives = 5
            lives = 5
            frames = 0
            if (epsilon > eps_min and episode % 100 == 0):
                epsilon = epsilon-1
                print('epsilon--')
                print(epsilon)
            episode += 1

            # Backpropagate
            gradients = np.vstack(agent.gradients)
            rewards = np.vstack(agent.rewards)

            rewards = agent.discount_rewards(rewards)
            if normalize:
                std = np.std(rewards - np.mean(rewards))
                rewards = rewards / std if std > 0 else 1
            gradients *= rewards
            X = np.squeeze(np.vstack([agent.states]))
            Y = agent.probs + agent.learning_rate * np.squeeze(np.vstack([gradients]))

            agent.model.train_on_batch(X, Y)
            agent.states, agent.probs, agent.gradients, agent.rewards = [], [], [], []

            print('Episode: %d - Score: %f - Real reward: %f - Avg Reward: %f' % (episode, score, real_reward, avg_reward))
            f.write("%d, %f, %f\n" % (episode, score, avg_reward))
            score = 0
            real_reward = 0
            state = env.reset()
            prev_x = None
            if episode > 1 and episode % 10 == 0:
                agent.save('forced shoot reward structure 1.h5')