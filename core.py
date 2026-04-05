# core.py
# 智学导航核心算法模块（与界面无关）

import numpy as np
from typing import List, Dict, Any

# ============================================================
# 扩散模型与不确定性估计
# ============================================================

class ConditionedDiffusionModel:
    def __init__(self, noise_scale=0.08):
        self.noise_scale = noise_scale

    def sample_states(self, current_state, num_samples=20):
        samples = []
        for _ in range(num_samples):
            noise = np.random.randn(len(current_state)) * self.noise_scale
            sample = np.clip(current_state + noise, 0.05, 0.95)
            samples.append(sample)
        return samples


class UncertaintyEstimator:
    @staticmethod
    def estimate_uncertainty(state_samples):
        if not state_samples:
            return 0.0
        states = np.array(state_samples)
        return np.mean(np.var(states, axis=0))


# ============================================================
# 模拟学生环境
# ============================================================

class SimulatedStudent:
    def __init__(self, num_concepts=6, seed=None, init_std=0.03,
                 use_paper_params=False,
                 learning_rate=0.16, forget_factor=0.18):
        self.num_concepts = num_concepts
        if seed is not None:
            np.random.seed(seed)

        base_difficulties = [0.3, 0.5, 0.7, 0.8, 0.9]
        self.difficulties = np.array([
            base_difficulties[i % len(base_difficulties)]
            for i in range(num_concepts)
        ])

        if use_paper_params:
            self.learning_rate = 0.18
            self.forget_factor = 0.16
        else:
            self.learning_rate = learning_rate
            self.forget_factor = forget_factor

        self.init_std = init_std
        self.mastery = np.random.normal(0.35, init_std, num_concepts)
        self.mastery = np.clip(self.mastery, 0.25, 0.45)
        self.initial_mastery = self.mastery.copy()
        self.history = {
            'mastery': [self.mastery.copy()],
            'actions': [],
            'rewards': [],
        }
        self.current_step = 0
        self.max_steps = 45

    def get_state(self):
        return {
            'mastery': self.mastery.copy(),
            'difficulties': self.difficulties.copy(),
            'concept_ids': np.arange(self.num_concepts).tolist()
        }

    def reset(self):
        self.mastery = np.random.normal(0.35, self.init_std, self.num_concepts)
        self.mastery = np.clip(self.mastery, 0.25, 0.45)
        self.initial_mastery = self.mastery.copy()
        self.history = {
            'mastery': [self.mastery.copy()],
            'actions': [],
            'rewards': [],
        }
        self.current_step = 0
        return self.get_state()

    def step(self, action):
        if self.current_step >= self.max_steps:
            return self.get_state(), 0.0, True, {}

        mastery_before = self.mastery.copy()
        difficulty = self.difficulties[action]

        success_prob = mastery_before[action] * (1 - difficulty)
        success_prob = np.clip(success_prob, 0.1, 0.85)
        success = np.random.random() < success_prob

        if success:
            learning_gain = self.learning_rate * (1 - mastery_before[action]) * (1 - difficulty)
            if difficulty > 0.7:
                learning_gain *= 0.8
            self.mastery[action] = min(0.95, mastery_before[action] + learning_gain)

            base_reward = 1.2
            mastery_reward = np.exp(learning_gain * 4) * 3.0
            difficulty_bonus = difficulty * 3.0
            reward = base_reward + mastery_reward + difficulty_bonus
            reward *= np.random.uniform(0.92, 1.08)
        else:
            penalty = self.forget_factor * mastery_before[action] * difficulty
            self.mastery[action] = max(0.05, mastery_before[action] - penalty)
            reward = -1.8

        self.history['mastery'].append(self.mastery.copy())
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.get_state(), reward, done, {}


# ============================================================
# 策略基类与具体策略
# ============================================================

class BaseStrategy:
    def __init__(self, name):
        self.name = name
        self.num_concepts = 6
        self.history = {'actions': [], 'rewards': [], 'observations': []}

    def select_action(self, state):
        raise NotImplementedError

    def update(self, action, reward, next_state):
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        obs = 1 if reward > 0 else 0
        self.history['observations'].append(obs)

    def reset(self):
        self.history = {'actions': [], 'rewards': [], 'observations': []}


class UA_MPCStrategy(BaseStrategy):
    def __init__(self, student_env, uncertainty_weight=0.5, diffusion_noise_scale=0.08, paper_mode=False):
        super().__init__(name="UA-MPC")
        if paper_mode:
            self.planning_horizon = 5
            self.uncertainty_weight = uncertainty_weight
            self.diffusion_model = ConditionedDiffusionModel(noise_scale=0.08)
            self.uncertainty_estimator = UncertaintyEstimator()
        else:
            self.planning_horizon = 8
            self.uncertainty_weight = uncertainty_weight
            self.diffusion_model = ConditionedDiffusionModel(noise_scale=diffusion_noise_scale)
            self.uncertainty_estimator = UncertaintyEstimator()

        self.env_learning_rate = student_env.learning_rate
        self.env_forget_factor = student_env.forget_factor
        self.reset()

    def select_action(self, state):
        samples = self.diffusion_model.sample_states(state['mastery'], num_samples=20)
        candidates = self._generate_candidates(state)
        best_action = None
        best_score = -float('inf')
        for action in candidates:
            score = self._evaluate_action_with_uncertainty(action, samples, state)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action if best_action is not None else np.argmin(state['mastery'])

    def _generate_candidates(self, state):
        mastery = state['mastery']
        gaps = 1 - mastery
        success_rates = self._calculate_success_rate_vector()
        teaching_freq = self._calculate_teaching_frequency()
        scores = gaps * 1.5 + success_rates * 0.8 + (1 - teaching_freq) * 0.6
        top_k = min(8, self.num_concepts)
        top_concepts = np.argsort(scores)[-top_k:]
        return list(top_concepts)

    def _evaluate_action_with_uncertainty(self, action, samples, state):
        total_reward = 0
        total_uncertainty = 0
        difficulty = state['difficulties'][action]
        for s in samples:
            current_state = s.copy()
            for step in range(self.planning_horizon):
                next_state, reward = self._predict_next_state(current_state, action, difficulty)
                uncertainty = self.uncertainty_estimator.estimate_uncertainty([next_state])
                total_reward += reward * (0.95 ** step)
                total_uncertainty += uncertainty
                current_state = next_state
        avg_reward = total_reward / len(samples)
        avg_uncertainty = total_uncertainty / len(samples)
        return avg_reward - self.uncertainty_weight * avg_uncertainty

    def _predict_next_state(self, current_state, action, difficulty):
        mastery = current_state[action]
        p_success = mastery * (1 - difficulty)
        gain = self.env_learning_rate * (1 - mastery) * (1 - difficulty)
        next_mastery_success = min(0.95, mastery + gain)
        reward_success = 1.2 + np.exp(gain * 4) * 3.0 + difficulty * 3.0
        penalty = self.env_forget_factor * mastery * difficulty
        next_mastery_fail = max(0.05, mastery - penalty)
        reward_fail = -1.8
        expected_reward = p_success * reward_success + (1 - p_success) * reward_fail
        expected_next_mastery = p_success * next_mastery_success + (1 - p_success) * next_mastery_fail
        next_state = current_state.copy()
        next_state[action] = expected_next_mastery
        return next_state, expected_reward

    def _calculate_success_rate(self, concept_idx):
        if len(self.history['actions']) == 0:
            return 0.5
        correct = 0
        total = 0
        for i, act in enumerate(self.history['actions']):
            if act == concept_idx and i < len(self.history['observations']) - 1:
                obs = self.history['observations'][i + 1]
                if obs is not None:
                    total += 1
                    if obs == 1:
                        correct += 1
        return correct / total if total > 0 else 0.5

    def _calculate_success_rate_vector(self):
        return np.array([self._calculate_success_rate(i) for i in range(self.num_concepts)])

    def _calculate_teaching_frequency(self):
        if len(self.history['actions']) == 0:
            return np.zeros(self.num_concepts)
        freq = np.zeros(self.num_concepts)
        for act in self.history['actions']:
            freq[act] += 1
        return freq / len(self.history['actions'])


class MPC_NoUncertaintyStrategy(BaseStrategy):
    def __init__(self, student_env, paper_mode=False):
        super().__init__(name="无不确定性 MPC")
        if paper_mode:
            self.planning_horizon = 3
            self.random_action_prob = 0.15
            self.diffusion_model = ConditionedDiffusionModel(noise_scale=0.1)
            self.samples_num = 12
        else:
            self.planning_horizon = 8
            self.random_action_prob = 0.0
            self.diffusion_model = ConditionedDiffusionModel(noise_scale=0.08)
            self.samples_num = 20

        self.env_learning_rate = student_env.learning_rate
        self.env_forget_factor = student_env.forget_factor
        self.history['states'] = []

    def select_action(self, state):
        self.history['states'].append(state['mastery'].copy())
        if np.random.random() < self.random_action_prob:
            return np.random.randint(0, self.num_concepts)

        current_state = state['mastery']
        samples = self.diffusion_model.sample_states(current_state, num_samples=self.samples_num)
        best_action = None
        best_score = -float('inf')
        for action in range(self.num_concepts):
            total_score = 0
            difficulty = state['difficulties'][action]
            for s in samples:
                score = self._evaluate_action(s, action, difficulty)
                total_score += score
            avg_score = total_score / len(samples)
            if avg_score > best_score:
                best_score = avg_score
                best_action = action
        return best_action if best_action is not None else np.argmin(state['mastery'])

    def _evaluate_action(self, start_state, action, difficulty):
        total_reward = 0
        discount = 0.95
        state = start_state.copy()
        for step in range(self.planning_horizon):
            mastery = state[action]
            p_success = mastery * (1 - difficulty)
            gain = self.env_learning_rate * (1 - mastery) * (1 - difficulty)
            next_mastery_success = min(0.95, mastery + gain)
            reward_success = 1.2 + np.exp(gain * 4) * 3.0 + difficulty * 3.0
            penalty = self.env_forget_factor * mastery * difficulty
            next_mastery_fail = max(0.05, mastery - penalty)
            reward_fail = -1.8
            expected_reward = p_success * reward_success + (1 - p_success) * reward_fail
            expected_next_mastery = p_success * next_mastery_success + (1 - p_success) * next_mastery_fail
            total_reward += expected_reward * (discount ** step)
            state[action] = expected_next_mastery
        return total_reward

    def update(self, action, reward, next_state):
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        obs = 1 if reward > 0 else 0
        self.history['observations'].append(obs)
        self.history['states'].append(next_state['mastery'].copy())

    def reset(self):
        self.history = {'actions': [], 'rewards': [], 'observations': [], 'states': []}


class SimpleEffectiveStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="SimpleEffective")

    def select_action(self, state):
        return np.argmin(state['mastery'])


class RandomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="随机策略")

    def select_action(self, state):
        return np.random.randint(0, self.num_concepts)


class DQNStrategy(BaseStrategy):
    def __init__(self, paper_mode=False):
        super().__init__(name="DQN")
        self.num_concepts = 6
        self.state_bins = 5
        self.state_shape = tuple([self.state_bins] * self.num_concepts)
        self.q_table = np.random.randn(*self.state_shape + (self.num_concepts,)) * 0.1
        self.learning_rate = 0.2
        self.discount_factor = 0.85
        if paper_mode:
            self.epsilon = 0.35
            self.min_epsilon = 0.15
            self.epsilon_decay = 0.995
        else:
            self.epsilon = 0.0
            self.min_epsilon = 0.0
            self.epsilon_decay = 1.0

        self.recent_experience = []
        self.max_experience = 50
        self.last_state = None
        self.last_action = None

    def _discretize_state(self, state_dict):
        mastery = state_dict['mastery']
        discrete_state = []
        for m in mastery:
            bin_idx = min(int(m * self.state_bins), self.state_bins - 1)
            discrete_state.append(bin_idx)
        return tuple(discrete_state)

    def select_action(self, state):
        discrete_state = self._discretize_state(state)
        if np.random.random() < self.epsilon:
            mastery = state['mastery']
            weights = 1.0 - np.array(mastery)
            weights = weights / np.sum(weights)
            action = np.random.choice(self.num_concepts, p=weights)
        else:
            q_values = self.q_table[discrete_state]
            action = np.argmax(q_values)
        self.last_state = discrete_state
        self.last_action = action
        return action

    def update(self, action, reward, next_state):
        next_discrete_state = self._discretize_state(next_state)
        self.recent_experience.append({
            'state': self.last_state,
            'action': action,
            'reward': reward,
            'next_state': next_discrete_state
        })
        if len(self.recent_experience) > self.max_experience:
            self.recent_experience.pop(0)

        current_q = self.q_table[self.last_state][action]
        max_next_q = np.max(self.q_table[next_discrete_state])
        target_q = reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[self.last_state][action] = new_q

        if len(self.recent_experience) >= 10 and np.random.random() < 0.3:
            self._experience_replay()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _experience_replay(self):
        batch_size = min(8, len(self.recent_experience))
        indices = np.random.choice(len(self.recent_experience), batch_size, replace=False)
        for idx in indices:
            exp = self.recent_experience[idx]
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            current_q = self.q_table[state][action]
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.q_table[state][action] = new_q

    def reset(self):
        self.q_table = np.random.randn(*self.q_table.shape) * 0.1
        self.epsilon = 0.35
        self.recent_experience = []
        self.last_state = None
        self.last_action = None
        super().reset()


class BKTThompsonStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="BKT-Thompson")
        self.alpha = np.ones(self.num_concepts) * 2.0
        self.beta = np.ones(self.num_concepts) * 2.0

    def select_action(self, state):
        sampled_values = np.zeros(self.num_concepts)
        for i in range(self.num_concepts):
            sampled_values[i] = np.random.beta(self.alpha[i], self.beta[i])
        return np.argmax(sampled_values)

    def update(self, action, reward, next_state):
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

    def reset(self):
        self.alpha = np.ones(self.num_concepts) * 2.0
        self.beta = np.ones(self.num_concepts) * 2.0


class IRTStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="IRT")
        self.ability_estimates = np.zeros(self.num_concepts) + 0.5
        self.discrimination = 1.0

    def select_action(self, state):
        return np.argmax(self.ability_estimates)

    def update(self, action, reward, next_state):
        step = 0.1
        if reward > 0:
            self.ability_estimates[action] += step * (1 - self.ability_estimates[action])
        else:
            self.ability_estimates[action] -= step * self.ability_estimates[action]
        self.ability_estimates[action] = np.clip(self.ability_estimates[action], 0.1, 0.9)

    def reset(self):
        self.ability_estimates = np.zeros(self.num_concepts) + 0.5


class LightweightDKTStateTracker:
    def __init__(self, n_concepts, hidden_dim=32):
        self.n_concepts = n_concepts
        self.hidden_dim = hidden_dim
        input_dim = n_concepts * 3
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, n_concepts) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(n_concepts)

    def forward(self, x):
        h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)
        h1 = h1 * (np.random.rand(*h1.shape) > 0.1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        h2 = h2 * (np.random.rand(*h2.shape) > 0.1)
        logits = np.dot(h2, self.W3) + self.b3
        pred = 1.0 / (1.0 + np.exp(-logits))
        return pred, h2

    def extract_deep_features(self, history):
        if len(history.get('states', [])) == 0:
            return np.zeros(self.n_concepts * 3)

        current_state = history['states'][-1]
        concept_freq = np.zeros(self.n_concepts)
        concept_success = np.zeros(self.n_concepts)
        concept_counts = np.zeros(self.n_concepts)

        for i, action in enumerate(history.get('actions', [])):
            concept = action
            concept_freq[concept] += 1
            if i < len(history.get('observations', [])) - 1:
                obs = history['observations'][i + 1]
                if obs is not None:
                    concept_success[concept] += obs
                    concept_counts[concept] += 1

        if len(history.get('actions', [])) > 0:
            concept_freq = concept_freq / len(history['actions'])
        else:
            concept_freq = np.ones(self.n_concepts) / self.n_concepts

        concept_success_rate = np.zeros(self.n_concepts)
        for i in range(self.n_concepts):
            if concept_counts[i] > 0:
                concept_success_rate[i] = concept_success[i] / concept_counts[i]
            else:
                concept_success_rate[i] = 0.5

        features = np.concatenate([current_state, concept_freq, concept_success_rate])
        target_len = self.n_concepts * 3
        if len(features) < target_len:
            features = np.pad(features, (0, target_len - len(features)))
        else:
            features = features[:target_len]
        return features


class DKTStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="DKT")
        self.dkt_tracker = LightweightDKTStateTracker(self.num_concepts)
        self.history = {'states': [], 'actions': [], 'observations': [], 'rewards': []}

    def select_action(self, state):
        self.history['states'].append(state['mastery'].copy())
        features = self.dkt_tracker.extract_deep_features(self.history)
        pred, _ = self.dkt_tracker.forward(features)
        return np.argmax(pred)

    def update(self, action, reward, next_state):
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        obs = 1 if reward > 0 else 0
        self.history['observations'].append(obs)
        self.history['states'].append(next_state['mastery'].copy())

    def reset(self):
        self.history = {'states': [], 'actions': [], 'observations': [], 'rewards': []}


# ============================================================
# 辅助函数（与界面无关）
# ============================================================

def auto_run_strategy(strategy, student, current_history, start_step):
    new_history = current_history.copy()
    step = start_step
    while step < 45:
        state = student.get_state()
        concept_names = [f'概念{i+1}' for i in range(6)]
        action = strategy.select_action(state)
        action_name = concept_names[action]
        next_state, reward, done, _ = student.step(action)
        strategy.update(action, reward, next_state)

        if hasattr(strategy, 'diffusion_model'):
            samples = strategy.diffusion_model.sample_states(state['mastery'], num_samples=30)
            uncertainty = strategy.uncertainty_estimator.estimate_uncertainty(samples) if hasattr(strategy, 'uncertainty_estimator') else 0.0
        else:
            uncertainty = 0.0

        step_info = {
            'step': step,
            'strategy': strategy.name,
            'action': action,
            'action_name': action_name,
            'reward': reward,
            'knowledge_mean': np.mean(next_state['mastery']),
            'uncertainty': uncertainty
        }
        new_history.append(step_info)
        step += 1
        if done:
            break
    return new_history, step


def compute_uncertainty(mastery, noise_scale=0.08, num_samples=30):
    temp_model = ConditionedDiffusionModel(noise_scale=noise_scale)
    samples = temp_model.sample_states(mastery, num_samples=num_samples)
    return UncertaintyEstimator.estimate_uncertainty(samples)