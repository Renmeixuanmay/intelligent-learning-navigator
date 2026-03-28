"""
智学导航：基于不确定性感知的个性化学习路径规划系统 (UA-MPC)
演示程序 - 完整版（包含 BKT-Thompson、IRT 贪心等基线）
（一键对比运行10次，表格居中，移除成功列）
增强版：不确定性热力图、数据导出、教师提示语
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(__file__)
font_path = os.path.join(current_dir, 'wqy-zenhei.ttc')

# 检查字体文件是否存在，若存在则加载
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    # 备选方案：使用系统可能预装的字体（仅当嵌入字体不可用时）
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 核心算法类定义
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


# ============================================================
# 新增基线：BKT-Thompson（贝叶斯知识追踪 + Thompson 采样）
# ============================================================
class BKTThompsonStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="BKT-Thompson")
        # 每个概念独立的 Beta 分布参数 (alpha, beta)，初始先验为 Beta(2,2) 即均值0.5
        self.alpha = np.ones(self.num_concepts) * 2.0
        self.beta = np.ones(self.num_concepts) * 2.0

    def select_action(self, state):
        # 从每个概念的 Beta 分布中采样一个掌握度估计，选择采样值最大的概念
        sampled_values = np.zeros(self.num_concepts)
        for i in range(self.num_concepts):
            sampled_values[i] = np.random.beta(self.alpha[i], self.beta[i])
        return np.argmax(sampled_values)

    def update(self, action, reward, next_state):
        # 根据答题结果更新 Beta 分布参数：成功则 alpha+1，失败则 beta+1
        if reward > 0:  # 成功
            self.alpha[action] += 1
        else:  # 失败
            self.beta[action] += 1

    def reset(self):
        self.alpha = np.ones(self.num_concepts) * 2.0
        self.beta = np.ones(self.num_concepts) * 2.0


# ============================================================
# 新增基线：IRT + 贪心策略（项目反应理论，选择匹配度最高的概念）
# ============================================================
class IRTGreedyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="IRT + 贪心")
        # 每个概念的难度估计（使用学生环境的实际难度，但策略内部可维护自己的估计）
        # 这里我们直接使用环境中的难度（因为真实难度已知）
        # 但为了演示，我们维护一个动态的能力估计（初始为0.5）
        self.ability_estimates = np.zeros(self.num_concepts) + 0.5  # 每个概念的能力估计
        self.discrimination = 1.0  # 区分度固定为1

    def select_action(self, state):
        # 使用当前能力估计和概念难度计算匹配度（答题成功概率）
        # 选择匹配度最高的概念（即最可能成功且信息量大的）
        # 这里使用 IRT 的 2PL 模型：P = 1 / (1 + exp(-discrimination * (ability - difficulty)))
        difficulties = state['difficulties']
        match_scores = np.zeros(self.num_concepts)
        for i in range(self.num_concepts):
            # 使用当前能力估计与难度计算期望正确概率
            z = self.discrimination * (self.ability_estimates[i] - difficulties[i])
            p = 1.0 / (1.0 + np.exp(-z))
            # 匹配度越高，说明学生与概念越匹配（既不过难也不过易）
            # 这里我们可以直接用概率作为匹配度，但为避免极端，也可用 -|ability - difficulty|
            # 采用 p 作为匹配度（越接近 0.5 时信息量最大，但这里选择高成功概率的概念）
            match_scores[i] = p
        return np.argmax(match_scores)

    def update(self, action, reward, next_state):
        # 根据答题结果更新该概念的能力估计（使用简单的增量更新）
        # 如果成功，能力估计增加；如果失败，能力估计降低
        step = 0.1  # 学习率
        if reward > 0:
            self.ability_estimates[action] += step * (1 - self.ability_estimates[action])
        else:
            self.ability_estimates[action] -= step * self.ability_estimates[action]
        # 限制在 [0.1, 0.9]
        self.ability_estimates[action] = np.clip(self.ability_estimates[action], 0.1, 0.9)

    def reset(self):
        self.ability_estimates = np.zeros(self.num_concepts) + 0.5


# ============================================================
# 辅助函数：自动运行策略直到45步
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


# ============================================================
# Streamlit 应用界面
# ============================================================

st.set_page_config(page_title="智学导航 · UA-MPC 演示", layout="wide")
st.title("🧠 智学导航：基于不确定性感知的个性化学习路径规划系统 (UA-MPC)")

# 添加自定义 CSS 使表格所有单元格居中
st.markdown("""
<style>
    .stDataFrame table td, .stDataFrame table th {
        text-align: center !important;
    }
    .stDataFrame table td[data-testid="stDataFrameCell"], 
    .stDataFrame table th[data-testid="stDataFrameHeaderCell"] {
        text-align: center !important;
    }
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        text-align: center !important;
    }
    .stDataFrame [data-testid="StyledDataFrameHeaderCell"] {
        text-align: center !important;
    }
    .stDataFrame .dataframe td, .stDataFrame .dataframe th {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)


# 初始化 session_state
if 'student' not in st.session_state:
    st.session_state.student = None
if 'strategy_ua' not in st.session_state:
    st.session_state.strategy_ua = None
if 'strategy_compare' not in st.session_state:
    st.session_state.strategy_compare = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0
if 'done' not in st.session_state:
    st.session_state.done = False
if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False
if 'compare_type' not in st.session_state:
    st.session_state.compare_type = "无不确定性 MPC"
if 'use_paper_params' not in st.session_state:
    st.session_state.use_paper_params = False

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 实验参数")
    seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)

    use_paper = st.checkbox("使用论文参数 (η=0.18, γ=0.16, 策略参数与论文一致)", value=st.session_state.use_paper_params)
    if use_paper != st.session_state.use_paper_params:
        st.session_state.use_paper_params = use_paper
        st.session_state.student = None
        st.session_state.strategy_ua = None
        st.session_state.strategy_compare = None
        st.session_state.history = []
        st.session_state.step_count = 0
        st.session_state.done = False
        st.rerun()

    if use_paper:
        st.info("当前使用论文固定参数：学习率=0.18，遗忘因子=0.16；UA-MPC 视野=5，λ=0.5；无不确定性 MPC 视野=3，随机探索概率0.15；DQN 初始ε=0.35")
        env_lr = 0.18
        env_forget = 0.16
        difficulty_level = None
        default_lambda = 0.5
        default_noise = 0.08
    else:
        difficulty_level = st.slider("环境难度", 0.0, 1.0, 0.8, 0.05)
        env_lr = 0.20 - difficulty_level * 0.10
        env_forget = 0.10 + difficulty_level * 0.15
        st.caption(f"当前学习率: {env_lr:.2f}  遗忘因子: {env_forget:.2f}")
        default_lambda = 0.6
        default_noise = 0.08

    lambda_uw = st.slider("不确定性权重 λ (UA-MPC)", 0.0, 1.0, default_lambda, 0.05)
    noise_scale = st.slider("扩散噪声尺度", 0.02, 0.2, default_noise, 0.01)

    st.divider()
    st.header("🔀 对比模式")
    compare_on = st.checkbox("开启策略对比", value=st.session_state.compare_mode)
    if compare_on:
        # 更新对比策略列表，包含新增的 BKT-Thompson 和 IRT+贪心
        compare_type = st.selectbox(
            "选择对比策略",
            ["无不确定性 MPC", "SimpleEffective", "DQN", "随机策略", "BKT-Thompson", "IRT + 贪心"],
            index=0
        )
    else:
        compare_type = None

    if (compare_on != st.session_state.compare_mode) or (compare_type != st.session_state.compare_type):
        st.session_state.compare_mode = compare_on
        st.session_state.compare_type = compare_type
        st.session_state.student = None
        st.session_state.strategy_ua = None
        st.session_state.strategy_compare = None
        st.session_state.history = []
        st.session_state.step_count = 0
        st.session_state.done = False
        st.rerun()

    st.divider()

    # 一键对比按钮（运行10次）
    if st.session_state.compare_mode and st.session_state.strategy_compare is not None:
        if st.button("⚡ 一键对比（UA-MPC vs " + st.session_state.compare_type + "）", use_container_width=True):
            runs = 10  # 运行10次
            all_hist = []
            ua_curves = []
            comp_curves = []
            seeds = [seed + i for i in range(runs)]

            with st.spinner(f"正在运行 {runs} 次实验..."):
                try:
                    for s in seeds:
                        # UA-MPC
                        student1 = SimulatedStudent(
                            num_concepts=6, seed=s, init_std=0.03,
                            use_paper_params=use_paper,
                            learning_rate=env_lr, forget_factor=env_forget
                        )
                        strategy1 = UA_MPCStrategy(
                            student_env=student1,
                            uncertainty_weight=lambda_uw,
                            diffusion_noise_scale=noise_scale,
                            paper_mode=use_paper
                        )
                        hist1, _ = auto_run_strategy(strategy1, student1, [], 0)
                        ua_curves.append([step['knowledge_mean'] for step in hist1])
                        all_hist.extend(hist1)

                        # 对比策略
                        student2 = SimulatedStudent(
                            num_concepts=6, seed=s, init_std=0.03,
                            use_paper_params=use_paper,
                            learning_rate=env_lr, forget_factor=env_forget
                        )
                        if st.session_state.compare_type == "无不确定性 MPC":
                            strategy2 = MPC_NoUncertaintyStrategy(student_env=student2, paper_mode=use_paper)
                        elif st.session_state.compare_type == "SimpleEffective":
                            strategy2 = SimpleEffectiveStrategy()
                        elif st.session_state.compare_type == "DQN":
                            strategy2 = DQNStrategy(paper_mode=use_paper)
                        elif st.session_state.compare_type == "随机策略":
                            strategy2 = RandomStrategy()
                        elif st.session_state.compare_type == "BKT-Thompson":
                            strategy2 = BKTThompsonStrategy()
                        elif st.session_state.compare_type == "IRT + 贪心":
                            strategy2 = IRTGreedyStrategy()
                        else:
                            strategy2 = None
                        hist2, _ = auto_run_strategy(strategy2, student2, [], 0)
                        comp_curves.append([step['knowledge_mean'] for step in hist2])
                        all_hist.extend(hist2)

                except Exception as e:
                    st.error(f"运行出错: {e}")
                    st.stop()

            # 计算均值和标准差
            ua_array = np.array(ua_curves)
            comp_array = np.array(comp_curves)
            ua_mean = np.mean(ua_array, axis=0)
            ua_std = np.std(ua_array, axis=0)
            comp_mean = np.mean(comp_array, axis=0)
            comp_std = np.std(comp_array, axis=0)

            # 绘制平均曲线
            fig, ax = plt.subplots(figsize=(12, 5))
            steps = np.arange(len(ua_mean))
            ax.plot(steps, ua_mean, 'o-', color='#4C72B0', label='UA-MPC')
            ax.fill_between(steps, ua_mean - ua_std, ua_mean + ua_std, color='#4C72B0', alpha=0.2)
            ax.plot(steps, comp_mean, 's-', color='#C44E52', label=st.session_state.compare_type)
            ax.fill_between(steps, comp_mean - comp_std, comp_mean + comp_std, color='#C44E52', alpha=0.2)
            ax.set_xlabel("教学步数", fontsize=12)
            ax.set_ylabel("平均知识水平", fontsize=12)
            ax.set_ylim(0.2, 0.8)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.success(f"运行 {runs} 次完成")
            st.write(f"**UA-MPC 最终知识水平**：{ua_mean[-1]:.3f} ± {ua_std[-1]:.3f}")
            st.write(f"**{st.session_state.compare_type} 最终知识水平**：{comp_mean[-1]:.3f} ± {comp_std[-1]:.3f}")

            st.session_state.history = all_hist
            st.session_state.step_count = 45
            st.session_state.done = True
            st.session_state.student = None
            st.session_state.strategy_ua = None
            st.session_state.strategy_compare = None

    st.divider()
    if st.button("🔄 重置学生", use_container_width=True):
        st.session_state.student = SimulatedStudent(
            num_concepts=6, seed=seed, init_std=0.03,
            use_paper_params=use_paper,
            learning_rate=env_lr, forget_factor=env_forget
        )
        st.session_state.strategy_ua = UA_MPCStrategy(
            student_env=st.session_state.student,
            uncertainty_weight=lambda_uw,
            diffusion_noise_scale=noise_scale,
            paper_mode=use_paper
        )
        if st.session_state.compare_mode:
            if st.session_state.compare_type == "无不确定性 MPC":
                st.session_state.strategy_compare = MPC_NoUncertaintyStrategy(
                    student_env=st.session_state.student,
                    paper_mode=use_paper
                )
            elif st.session_state.compare_type == "SimpleEffective":
                st.session_state.strategy_compare = SimpleEffectiveStrategy()
            elif st.session_state.compare_type == "DQN":
                st.session_state.strategy_compare = DQNStrategy(paper_mode=use_paper)
            elif st.session_state.compare_type == "随机策略":
                st.session_state.strategy_compare = RandomStrategy()
            elif st.session_state.compare_type == "BKT-Thompson":
                st.session_state.strategy_compare = BKTThompsonStrategy()
            elif st.session_state.compare_type == "IRT + 贪心":
                st.session_state.strategy_compare = IRTGreedyStrategy()
            else:
                st.session_state.strategy_compare = None
        else:
            st.session_state.strategy_compare = None
        st.session_state.history = []
        st.session_state.step_count = 0
        st.session_state.done = False
        st.rerun()

    st.divider()
    st.caption(f"当前步数: {st.session_state.step_count}/45")
    if st.session_state.done:
        st.warning("教学已完成 (45步)")

# 主面板布局
col_left, col_right = st.columns([2, 1.5])

with col_left:
    st.subheader("📊 当前知识状态 (含不确定性)")

    if st.session_state.student is not None:
        state = st.session_state.student.get_state()
        mastery = state['mastery']

        samples = st.session_state.strategy_ua.diffusion_model.sample_states(mastery, num_samples=30)
        samples_array = np.array(samples)
        mean_mastery = np.mean(samples_array, axis=0)
        std_mastery = np.std(samples_array, axis=0)
        overall_uncertainty = np.mean(std_mastery)

        # 条形图
        fig, ax = plt.subplots(figsize=(10, 5))
        concepts = [f'概念{i+1}' for i in range(6)]
        x = np.arange(len(concepts))
        bars = ax.bar(x, mean_mastery, yerr=std_mastery, capsize=8,
                      color='#4C72B0', ecolor='#C44E52', alpha=0.9,
                      error_kw={'linewidth': 2})
        ax.set_xticks(x)
        ax.set_xticklabels(concepts, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_ylabel("掌握程度", fontsize=12)
        ax.set_title("各概念掌握度均值 ± 不确定性 (标准差)", fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)

        # 不确定性热力图
        fig_heat, ax_heat = plt.subplots(figsize=(6, 2))
        heat_data = np.array([std_mastery])  # 1×6 矩阵
        im = ax_heat.imshow(heat_data, cmap='Reds', aspect='auto', vmin=0, vmax=0.25)
        ax_heat.set_xticks(np.arange(len(concepts)))
        ax_heat.set_xticklabels(concepts, fontsize=10)
        ax_heat.set_yticks([])
        ax_heat.set_title("各概念不确定性热力图（颜色越深，不确定性越高）", fontsize=12)
        plt.colorbar(im, ax=ax_heat, orientation='horizontal', pad=0.2, label='不确定性值')
        st.pyplot(fig_heat)

        # 整体不确定性指标
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("整体认知不确定性", f"{overall_uncertainty:.3f}")
        with col_m2:
            st.progress(min(overall_uncertainty / 0.3, 1.0), text="不确定性水平")
        with col_m3:
            st.metric("平均掌握度", f"{np.mean(mastery):.2f}")

        st.session_state.current_uncertainty = overall_uncertainty
    else:
        st.info("请先在侧边栏点击「重置学生」开始")

# ========== 右侧决策区域 ==========
with col_right:
    if st.session_state.student is not None and not st.session_state.done:
        state = st.session_state.student.get_state()
        concept_names = [f'概念{i+1}' for i in range(6)]

        # 教师提示语（当不确定性过高时）
        overall_uncertainty = st.session_state.get('current_uncertainty', 0.0)
        if overall_uncertainty > 0.1:
            st.warning("🧑‍🏫 教师提示：当前学生状态不确定性较高，建议教师给予针对性辅导。")

        # --- UA-MPC 决策卡片 ---
        with st.container(border=True):
            st.subheader("🎯 UA-MPC 决策")
            recommended_action_ua = st.session_state.strategy_ua.select_action(state)
            rec_name_ua = concept_names[recommended_action_ua]

            col_rec1, col_rec2, col_rec3 = st.columns([1, 1, 1])
            with col_rec1:
                st.metric("推荐概念", rec_name_ua, delta=None)
            with col_rec2:
                if st.button("✅ 执行", key="btn_ua", use_container_width=True):
                    action = recommended_action_ua
                    next_state, reward, done, _ = st.session_state.student.step(action)
                    st.session_state.strategy_ua.update(action, reward, next_state)

                    step_info = {
                        'step': st.session_state.step_count,
                        'strategy': 'UA-MPC',
                        'action': action,
                        'action_name': rec_name_ua,
                        'reward': reward,
                        'knowledge_mean': np.mean(next_state['mastery']),
                        'uncertainty': st.session_state.get('current_uncertainty', 0.0)
                    }
                    st.session_state.history.append(step_info)
                    st.session_state.step_count += 1
                    st.session_state.done = done
                    st.rerun()
            with col_rec3:
                if st.button("⚡ 自动45步", key="auto_ua", use_container_width=True):
                    new_hist, new_step = auto_run_strategy(
                        st.session_state.strategy_ua,
                        st.session_state.student,
                        st.session_state.history,
                        st.session_state.step_count
                    )
                    st.session_state.history = new_hist
                    st.session_state.step_count = new_step
                    st.session_state.done = (new_step >= 45)
                    st.rerun()

            with st.expander("查看选择依据"):
                mastery = state['mastery']
                difficulty = state['difficulties'][recommended_action_ua]
                st.write(f"- 掌握度: {mastery[recommended_action_ua]:.2f}")
                st.write(f"- 难度: {difficulty:.2f}")
                st.write(f"- 不确定性惩罚权重 λ={st.session_state.strategy_ua.uncertainty_weight:.2f}")

        # --- 对比策略决策卡片 (如果开启) ---
        if st.session_state.compare_mode and st.session_state.strategy_compare is not None:
            with st.container(border=True):
                st.subheader(f"🔄 {st.session_state.strategy_compare.name} 决策")
                recommended_action_comp = st.session_state.strategy_compare.select_action(state)
                rec_name_comp = concept_names[recommended_action_comp]

                col_rec1, col_rec2, col_rec3 = st.columns([1, 1, 1])
                with col_rec1:
                    st.metric("推荐概念", rec_name_comp, delta=None)
                with col_rec2:
                    if st.button(f"✅ 执行", key="btn_comp", use_container_width=True):
                        action = recommended_action_comp
                        next_state, reward, done, _ = st.session_state.student.step(action)
                        st.session_state.strategy_compare.update(action, reward, next_state)

                        step_info = {
                            'step': st.session_state.step_count,
                            'strategy': st.session_state.strategy_compare.name,
                            'action': action,
                            'action_name': rec_name_comp,
                            'reward': reward,
                            'knowledge_mean': np.mean(next_state['mastery']),
                            'uncertainty': st.session_state.get('current_uncertainty', 0.0)
                        }
                        st.session_state.history.append(step_info)
                        st.session_state.step_count += 1
                        st.session_state.done = done
                        st.rerun()
                with col_rec3:
                    if st.button(f"⚡ 自动45步", key="auto_comp", use_container_width=True):
                        new_hist, new_step = auto_run_strategy(
                            st.session_state.strategy_compare,
                            st.session_state.student,
                            st.session_state.history,
                            st.session_state.step_count
                        )
                        st.session_state.history = new_hist
                        st.session_state.step_count = new_step
                        st.session_state.done = (new_step >= 45)
                        st.rerun()

                with st.expander("查看选择依据"):
                    mastery = state['mastery']
                    difficulty = state['difficulties'][recommended_action_comp]
                    st.write(f"- 掌握度: {mastery[recommended_action_comp]:.2f}")
                    st.write(f"- 难度: {difficulty:.2f}")

    elif st.session_state.done:
        st.info("已达到最大步数，请重置学生重新开始。")
    else:
        st.info("请先重置学生")

# ========== 历史记录与学习曲线 ==========
st.divider()

st.subheader("📋 教学历史记录")
if st.session_state.history:
    # 导出CSV按钮
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 导出历史记录为CSV",
        data=csv,
        file_name="teaching_history.csv",
        mime="text/csv",
        use_container_width=True
    )
    # 显示表格
    df_display = df[['step', 'strategy', 'action_name', 'reward', 'knowledge_mean', 'uncertainty']].copy()
    df_display.columns = ['步数', '策略', '教学概念', '奖励', '平均知识水平', '不确定性']
    st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.caption("暂无历史记录")

if st.session_state.history:
    st.subheader("📈 平均知识水平学习曲线")
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    df_plot = pd.DataFrame(st.session_state.history)
    strategies = df_plot['strategy'].unique()
    colors = {
        'UA-MPC': '#4C72B0',
        '无不确定性 MPC': '#C44E52',
        'SimpleEffective': '#8172B2',
        '随机策略': '#2ca02c',
        'DQN': '#ff7f0e',
        'BKT-Thompson': '#9467bd',
        'IRT + 贪心': '#8c564b'
    }

    for strategy in strategies:
        df_strat = df_plot[df_plot['strategy'] == strategy]
        steps = df_strat['step'].values
        knowledge = df_strat['knowledge_mean'].values
        ax2.plot(steps, knowledge, 'o-', linewidth=2, markersize=6,
                 color=colors.get(strategy, '#000000'), label=strategy)

    ax2.set_xlabel("教学步数", fontsize=12)
    ax2.set_ylabel("平均知识水平", fontsize=12)
    ax2.set_ylim(0.2, 0.8)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

st.divider()
st.caption("""
**智学导航**：基于不确定性感知的教学规划系统 (UA-MPC)  
- 使用条件扩散模型生成知识状态分布，量化认知不确定性  
- 在模型预测控制中显式惩罚高不确定性，实现稳健教学决策  
- 支持论文固定参数模式（勾选后参数与论文实验脚本完全一致）  
- 一键对比功能：自动运行10次实验，绘制带误差带的平均学习曲线  
- 新增基线：BKT-Thompson、IRT+贪心（教育领域经典方法）  
- 新增功能：不确定性热力图、数据导出、教师提示语
""")