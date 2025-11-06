import random
import pickle
import csv


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    """
    @brief loads the Q state-action values from a pickle file.
    @param filename The name of the file (without extension) to load
    """
    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.

        try:
            with open(filename+".pickle", "rb") as f:
                self.q = pickle.load(f)
        except FileNotFoundError:
            print("File not found: {}".format(filename+".pickle"))
            self.q = {}

        print("Loaded file: {}".format(filename+".pickle"))

    """
    @brief saves the Q state-action values in a pickle file.
    @param filename The name of the file to save
    """
    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        with open(filename+".pickle", "wb") as f:
            pickle.dump(self.q, f)

        with open(filename+".csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['state', 'action', 'q_value'])
            for (state, action), q_value in self.q.items():
                writer.writerow([state, action, q_value])

        print("Wrote to file: {}".format(filename+".pickle"))

   
    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        @param state The current state
        @param return_q If True return action and q instead of just action
        @retval action or (action, q)
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            if return_q:
                return action, self.getQ(state, action)
            else:
                return action
        else:
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            chosen_action = random.choice(max_actions)
            if return_q:
                return chosen_action, max_q
            else:
                return chosen_action 


    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        @param state1 The current state
        @param action1 The action taken
        @param reward The reward received after taking action1 in state1
        @param state2 The next state after taking action1
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        current_q = self.getQ(state1, action1)

        future_q_values = [self.getQ(state2, a) for a in self.actions]
        max_future_q = max(future_q_values) if future_q_values else 0.0
        
        self.q[(state1, action1)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

      
        
