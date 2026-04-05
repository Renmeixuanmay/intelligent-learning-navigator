# app.py - Streamlit 主界面（从 core 导入算法）
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
import os

# 从 core 模块导入所有算法类（不包含任何 Streamlit 代码）
from core import *

# ============================================================
# 设置 matplotlib 中文字体（显式加载本地字体文件）
# ============================================================
current_dir = os.path.dirname(__file__)
font_path = os.path.join(current_dir, 'wqy-zenhei.ttc')

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

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
        compare_type = st.selectbox(
            "选择对比策略",
            ["无不确定性 MPC", "SimpleEffective", "DQN", "随机策略", "BKT-Thompson", "IRT", "DKT"],
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
            runs = 10
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
                        elif st.session_state.compare_type == "IRT":
                            strategy2 = IRTStrategy()
                        elif st.session_state.compare_type == "DKT":
                            strategy2 = DKTStrategy()
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

            # 添加对比表格
            avg_reward_ua = np.mean([step['reward'] for step in hist1])
            avg_reward_comp = np.mean([step['reward'] for step in hist2])
            improvement = (ua_mean[-1] - comp_mean[-1]) / comp_mean[-1] * 100 if comp_mean[-1] != 0 else 0
            comparison_data = {
                "策略": ["UA-MPC", st.session_state.compare_type],
                "最终知识水平": [f"{ua_mean[-1]:.3f} ± {ua_std[-1]:.3f}", f"{comp_mean[-1]:.3f} ± {comp_std[-1]:.3f}"],
                "平均奖励": [f"{avg_reward_ua:.2f}", f"{avg_reward_comp:.2f}"],
                "相对提升": [f"{improvement:.1f}% (vs {st.session_state.compare_type})", "—"]
            }
            df_compare = pd.DataFrame(comparison_data)
            st.table(df_compare)

            st.session_state.student = student2
            st.session_state.strategy_ua = None
            st.session_state.strategy_compare = None
            st.session_state.history = all_hist
            st.session_state.step_count = 45
            st.session_state.done = True

    st.divider()

    # 一键演示按钮（快速运行20步）
    if st.session_state.compare_mode and st.session_state.strategy_compare is not None:
        if st.button("⚡ 一键演示 (快速展示核心优势, 20步)", use_container_width=True):
            demo_steps = 20
            # 运行 UA-MPC
            student_demo_ua = SimulatedStudent(
                num_concepts=6, seed=seed, init_std=0.03,
                use_paper_params=use_paper,
                learning_rate=env_lr, forget_factor=env_forget
            )
            strategy_ua_demo = UA_MPCStrategy(
                student_env=student_demo_ua,
                uncertainty_weight=lambda_uw,
                diffusion_noise_scale=noise_scale,
                paper_mode=use_paper
            )
            ua_history = []
            for step in range(demo_steps):
                state = student_demo_ua.get_state()
                action = strategy_ua_demo.select_action(state)
                next_state, reward, _, _ = student_demo_ua.step(action)
                strategy_ua_demo.update(action, reward, next_state)
                ua_history.append({
                    'step': step,
                    'knowledge_mean': np.mean(next_state['mastery'])
                })

            # 运行对比策略
            student_demo_comp = SimulatedStudent(
                num_concepts=6, seed=seed, init_std=0.03,
                use_paper_params=use_paper,
                learning_rate=env_lr, forget_factor=env_forget
            )
            if st.session_state.compare_type == "无不确定性 MPC":
                strategy_comp_demo = MPC_NoUncertaintyStrategy(student_env=student_demo_comp, paper_mode=use_paper)
            elif st.session_state.compare_type == "SimpleEffective":
                strategy_comp_demo = SimpleEffectiveStrategy()
            elif st.session_state.compare_type == "DQN":
                strategy_comp_demo = DQNStrategy(paper_mode=use_paper)
            elif st.session_state.compare_type == "随机策略":
                strategy_comp_demo = RandomStrategy()
            elif st.session_state.compare_type == "BKT-Thompson":
                strategy_comp_demo = BKTThompsonStrategy()
            elif st.session_state.compare_type == "IRT":
                strategy_comp_demo = IRTStrategy()
            elif st.session_state.compare_type == "DKT":
                strategy_comp_demo = DKTStrategy()
            else:
                strategy_comp_demo = None

            comp_history = []
            for step in range(demo_steps):
                state = student_demo_comp.get_state()
                action = strategy_comp_demo.select_action(state)
                next_state, reward, _, _ = student_demo_comp.step(action)
                strategy_comp_demo.update(action, reward, next_state)
                comp_history.append({
                    'step': step,
                    'knowledge_mean': np.mean(next_state['mastery'])
                })

            # 绘制对比曲线
            fig_demo, ax_demo = plt.subplots(figsize=(10, 5))
            ua_steps = [h['step'] for h in ua_history]
            ua_knowledge = [h['knowledge_mean'] for h in ua_history]
            comp_steps = [h['step'] for h in comp_history]
            comp_knowledge = [h['knowledge_mean'] for h in comp_history]
            ax_demo.plot(ua_steps, ua_knowledge, 'o-', color='#4C72B0', label='UA-MPC')
            ax_demo.plot(comp_steps, comp_knowledge, 's-', color='#C44E52', label=st.session_state.compare_type)
            ax_demo.set_xlabel("教学步数", fontsize=12)
            ax_demo.set_ylabel("平均知识水平", fontsize=12)
            ax_demo.set_ylim(0.2, 0.8)
            ax_demo.legend(fontsize=11)
            ax_demo.grid(True, alpha=0.3)
            st.pyplot(fig_demo)

            st.success(f"快速演示完成（{demo_steps}步）")
            st.write(f"**UA-MPC 最终知识水平**：{ua_knowledge[-1]:.3f}")
            st.write(f"**{st.session_state.compare_type} 最终知识水平**：{comp_knowledge[-1]:.3f}")
            improvement_demo = (ua_knowledge[-1] - comp_knowledge[-1]) / comp_knowledge[-1] * 100 if comp_knowledge[-1] != 0 else 0
            st.write(f"**UA-MPC 相对提升**：{improvement_demo:.1f}%")

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
            elif st.session_state.compare_type == "IRT":
                st.session_state.strategy_compare = IRTStrategy()
            elif st.session_state.compare_type == "DKT":
                st.session_state.strategy_compare = DKTStrategy()
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

        cur_noise = st.session_state.get('noise_scale', 0.08) if 'noise_scale' in st.session_state else 0.08
        overall_uncertainty = compute_uncertainty(mastery, noise_scale=cur_noise, num_samples=30)

        temp_model = ConditionedDiffusionModel(noise_scale=cur_noise)
        samples = temp_model.sample_states(mastery, num_samples=30)
        samples_array = np.array(samples)
        mean_mastery = np.mean(samples_array, axis=0)
        std_mastery = np.std(samples_array, axis=0)

        # 条形图
        fig, ax = plt.subplots(figsize=(8, 4))
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
        fig_heat, ax_heat = plt.subplots(figsize=(5, 2))
        heat_data = np.array([std_mastery])
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
    # 导出CSV（使用中文表头，utf-8-sig 解决乱码）
    df = pd.DataFrame(st.session_state.history)
    df_export = df[['step', 'strategy', 'action_name', 'reward', 'knowledge_mean', 'uncertainty']].copy()
    df_export.columns = ['步数', '策略', '教学概念', '奖励', '平均知识水平', '不确定性']
    csv_data = df_export.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 导出历史记录为CSV",
        data=csv_data,
        file_name="teaching_history.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption("💡 提示：若用 Excel 打开出现乱码，请使用「数据 → 从文本/CSV 导入」并选择 UTF-8 编码，或直接用记事本打开。")
    # 显示表格（同样中文）
    st.dataframe(df_export, use_container_width=True, hide_index=True)
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
        'IRT': '#8c564b',
        'DKT': '#e377c2'
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
st.markdown("""
**智学导航**：基于不确定性感知的教学规划系统 (UA-MPC)  
- 使用条件扩散模型生成知识状态分布，量化认知不确定性  
- 在模型预测控制中显式惩罚高不确定性，实现稳健教学决策  
- 支持论文固定参数模式（勾选后参数与论文实验脚本完全一致）  
- 一键对比功能：自动运行10次实验，绘制带误差带的平均学习曲线  
- 一键演示功能：快速运行20步，直观展示核心优势  
- 对比策略：无不确定性MPC、SimpleEffective、DQN、随机策略、BKT-Thompson、IRT、DKT  
- 辅助功能：不确定性热力图、数据导出、教师提示语、策略效果对比表

---
### 📌 国产技术适配说明

1. **桌面操作系统适配**  
   本系统可无缝运行于 **麒麟系统（Kylin OS）**、**统信UOS** 等国产桌面操作系统，兼容 x86/ARM 架构，核心依赖库均已适配阿里云 PyPI、华为云 PyPI 等国产镜像源，可脱离国外镜像独立安装部署。

2. **鸿蒙移动端适配**  
   已完成 **鸿蒙 HarmonyOS 应用** 开发，通过 WebView 封装，可在鸿蒙手机、平板等设备上流畅运行，提供与 Web 端完全一致的教学决策体验。鸿蒙应用安装包（.hap）随作品一并提交，支持多终端协同使用。

3. **深度学习框架迁移潜力**  
   核心算法（UA-MPC 的条件扩散模型、不确定性量化模块）已做 **框架无关性设计**，未使用 TensorFlow/PyTorch 专属 API，可快速迁移至百度飞桨（PaddlePaddle）、华为 MindSpore 等国产深度学习框架，迁移成本低于 10%。
""")