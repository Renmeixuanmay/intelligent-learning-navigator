"""
智学导航 - QClaw 智能体 API 服务
为 QClaw 自定义 Skill 提供 HTTP 调用接口
"""

# api.py - QClaw 智能体 API 服务
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

# 从 core 导入所需算法类
from core import (
    SimulatedStudent,
    UA_MPCStrategy,
    ConditionedDiffusionModel,
    UncertaintyEstimator,
    compute_uncertainty
)

# 创建 FastAPI 应用
app = FastAPI(title="智学导航教学决策 API", description="为 QClaw 智能体提供 UA-MPC 教学概念推荐")

# 全局学生状态（简单内存存储，可扩展为基于用户 ID 的存储）
student_cache = {}


class TeachRequest(BaseModel):
    """QClaw 发来的教学决策请求"""
    query: str  # 用户自然语言指令（如“推荐一个教学概念”）
    user_id: Optional[str] = "default"  # 用户标识，用于区分不同学生
    concept_names: Optional[List[str]] = ["概念1", "概念2", "概念3", "概念4", "概念5", "概念6"]


class TeachResponse(BaseModel):
    """教学决策响应"""
    success: bool
    recommendation: str  # 推荐的教学概念名称
    concept_index: int  # 推荐的概念索引
    uncertainty: float  # 当前认知不确定性
    mastery_levels: List[float]  # 各概念掌握度
    message: str  # 返回给用户的可读消息


def get_or_create_student(user_id: str):
    """获取或创建学生实例（每个用户独立）"""
    if user_id not in student_cache:
        student_cache[user_id] = SimulatedStudent(
            num_concepts=6,
            seed=42,
            init_std=0.03,
            use_paper_params=False,
            learning_rate=0.18,
            forget_factor=0.16
        )
    return student_cache[user_id]


def get_or_create_strategy(student):
    """为每个学生创建对应的 UA-MPC 策略实例"""
    return UA_MPCStrategy(
        student_env=student,
        uncertainty_weight=0.6,
        diffusion_noise_scale=0.08,
        paper_mode=False
    )


@app.get("/")
def root():
    return {"message": "智学导航教学决策 API 服务运行中", "status": "ok"}


@app.post("/api/recommend", response_model=TeachResponse)
def recommend_concept(request: TeachRequest):
    """
    核心教学决策接口
    - 接收 QClaw 发来的自然语言指令
    - 调用 UA-MPC 算法推荐最佳教学概念
    - 返回推荐结果及当前认知状态
    """
    try:
        # 1. 获取学生实例（QClaw 可通过 user_id 区分不同用户）
        student = get_or_create_student(request.user_id)

        # 2. 获取学生当前知识状态
        state = student.get_state()
        mastery = state['mastery']
        difficulties = state['difficulties']

        # 3. 创建 UA-MPC 策略实例
        strategy = get_or_create_strategy(student)

        # 4. 调用核心决策算法
        recommended_action = strategy.select_action(state)
        rec_name = request.concept_names[recommended_action]

        # 5. 计算当前不确定性
        uncertainty = compute_uncertainty(mastery, noise_scale=0.08, num_samples=30)

        # 6. 构造友好返回消息
        mastery_text = "、".join([f"{name}:{mastery[i]:.2f}" for i, name in enumerate(request.concept_names)])
        message = (
            f"🎯 智学导航为您推荐教学概念：**{rec_name}**\n\n"
            f"📊 当前知识状态：\n{mastery_text}\n\n"
            f"⚠️ 认知不确定性：{uncertainty:.3f}\n\n"
            f"💡 决策依据：该概念当前掌握度较低({mastery[recommended_action]:.2f})，\n"
            f"   学习难度适中，是当前提升知识水平的最佳选择。"
        )

        return TeachResponse(
            success=True,
            recommendation=rec_name,
            concept_index=recommended_action,
            uncertainty=uncertainty,
            mastery_levels=mastery.tolist(),
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"教学决策失败: {str(e)}")


@app.post("/api/step")
def execute_teaching_step(request: TeachRequest):
    """
    执行一步教学（可选功能：接收推荐 + 模拟教学反馈）
    调用后，学生状态会更新，下次推荐基于新的掌握度
    """
    try:
        student = get_or_create_student(request.user_id)
        strategy = get_or_create_strategy(student)
        state = student.get_state()

        # 获取推荐动作
        action = strategy.select_action(state)

        # 执行教学步骤（模拟学生反馈）
        next_state, reward, done, _ = student.step(action)
        strategy.update(action, reward, next_state)

        # 计算新状态的不确定性
        uncertainty = compute_uncertainty(next_state['mastery'], noise_scale=0.08, num_samples=30)

        # 确定是否完成（45步上限）
        status = "已完成45步教学" if done else f"教学进行中（{student.current_step}/45步）"

        message = (
            f"✅ 已执行教学步骤：推荐概念 {request.concept_names[action]}\n"
            f"📈 教学反馈：{'正确理解 ✅' if reward > 0 else '需加强练习 ❌'}\n"
            f"📊 当前平均掌握度：{np.mean(next_state['mastery']):.3f}\n"
            f"⚠️ 不确定性：{uncertainty:.3f}\n"
            f"📌 {status}"
        )

        return {
            "success": True,
            "recommendation": request.concept_names[action],
            "reward": reward,
            "knowledge_mean": float(np.mean(next_state['mastery'])),
            "uncertainty": uncertainty,
            "step": student.current_step,
            "done": done,
            "message": message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行教学步骤失败: {str(e)}")


@app.get("/api/status")
def get_teaching_status(user_id: str = "default"):
    """获取当前教学状态（用于 QClaw 查询进度）"""
    student = get_or_create_student(user_id)
    state = student.get_state()
    uncertainty = compute_uncertainty(state['mastery'], noise_scale=0.08, num_samples=30)

    return {
        "success": True,
        "step": student.current_step,
        "max_steps": student.max_steps,
        "knowledge_mean": float(np.mean(state['mastery'])),
        "uncertainty": uncertainty,
        "mastery_levels": state['mastery'].tolist()
    }


@app.post("/api/reset")
def reset_teaching(user_id: str = "default"):
    """重置教学状态"""
    if user_id in student_cache:
        del student_cache[user_id]
    return {"success": True, "message": f"用户 {user_id} 的教学状态已重置"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)