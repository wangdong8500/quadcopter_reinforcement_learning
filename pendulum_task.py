import numpy as np
import gym 

class PendulumTask(object):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object.
        """
        # Simulation
        self.env = gym.make('Pendulum-v0')
        self.action_repeat = 3
        self.state_size = self.env.observation_space.shape[0] * self.action_repeat
        self.action_size = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward_total = 0
        state_all = []
        for _ in range(self.action_repeat):
            state, reward, done, info = self.env.step(action) # update the sim pose and velocities
            reward_total += reward
            state_all.append(state)
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.env.reset()
        state = np.concatenate([state] * self.action_repeat) 
        return state