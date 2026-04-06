from __future__ import annotations

import json
from textwrap import dedent

# ---------------------------------------------------------------------------
# Diagnostic criteria — injected dynamically only when hypothesis forms
# ---------------------------------------------------------------------------

DIAGNOSTIC_CRITERIA: dict[str, dict] = {
    "MDD": {
        "disorder_cn": "重性抑郁障碍",
        "criteria_A": {
            "required_count": 5,
            "duration": "连续至少2周",
            "must_include_one_of": ["depressed_mood", "anhedonia"],
            "items": {
                "depressed_mood": "几乎每天大部分时间情绪低落",
                "anhedonia": "对几乎所有活动兴趣或乐趣明显减少",
                "weight_change": "体重或食欲明显变化（增或减）",
                "sleep_disturbance": "失眠或睡眠过多",
                "psychomotor_change": "精神运动性激越或迟滞",
                "fatigue": "几乎每天疲劳或精力不足",
                "worthlessness_guilt": "无价值感或过度内疚",
                "concentration_difficulty": "思考或集中注意力困难",
                "suicidal_ideation": "反复出现死亡或自杀的想法",
            },
        },
        "exclusions": ["bipolar_history", "substance_induced", "medical_condition"],
    },
    "GAD": {
        "disorder_cn": "广泛性焦虑障碍",
        "criteria": {
            "core": "对多种事件或活动过度焦虑和担忧，持续至少6个月",
            "difficulty_controlling": "难以控制担忧",
            "items_3_of_6": {
                "restlessness": "坐立不安或感到紧绷",
                "fatigue": "容易疲劳",
                "concentration_difficulty": "注意力难以集中",
                "irritability": "易激惹",
                "muscle_tension": "肌肉紧张",
                "sleep_disturbance": "睡眠障碍（入睡困难或睡眠不安）",
            },
        },
        "exclusions": ["substance_induced", "medical_condition", "panic_disorder_primary"],
    },
    "schizophrenia": {
        "disorder_cn": "精神分裂症",
        "criteria_A": {
            "required_count": 2,
            "duration_active": "至少1个月",
            "duration_total": "至少6个月",
            "must_include_one_of": ["delusions", "hallucinations", "disorganized_speech"],
            "items": {
                "delusions": "妄想",
                "hallucinations": "幻觉",
                "disorganized_speech": "言语紊乱（思维松弛或联想散漫）",
                "disorganized_behavior": "明显紊乱或紧张症行为",
                "negative_symptoms": "阴性症状（情感平淡、言语贫乏、意志减退）",
            },
        },
        "exclusions": ["schizoaffective", "mood_disorder_primary", "substance_induced"],
    },
    "panic_disorder": {
        "disorder_cn": "惊恐障碍",
        "criteria": {
            "core": "反复发作的意外惊恐发作",
            "attack_symptoms_4_of_13": {
                "palpitations": "心悸或心跳加速",
                "sweating": "出汗",
                "trembling": "颤抖",
                "shortness_of_breath": "气短或窒息感",
                "choking": "哽咽感",
                "chest_pain": "胸痛或胸部不适",
                "nausea": "恶心或腹部不适",
                "dizziness": "头晕、步态不稳或晕厥感",
                "chills_hot": "发冷或发热",
                "paresthesia": "感觉异常（麻木或刺痛）",
                "derealization": "现实解体或人格解体",
                "fear_losing_control": "害怕失去控制或发疯",
                "fear_dying": "害怕死去",
            },
            "anticipatory_anxiety": "至少1次发作后持续担忧再次发作或改变行为",
        },
    },
    "adjustment_disorder": {
        "disorder_cn": "适应障碍",
        "criteria": {
            "core": "对可识别应激源的情绪或行为反应，在应激源出现3个月内发生",
            "disproportionate": "症状严重程度超出应激源本身预期",
            "duration": "应激源消除后不超过6个月",
        },
    },
}


def get_criteria_for_hypothesis(disorder: str) -> str:
    """Return formatted criteria string for injection into clinical_brain prompt."""
    if disorder not in DIAGNOSTIC_CRITERIA:
        return ""
    criteria = DIAGNOSTIC_CRITERIA[disorder]
    return f"\n[当前主要假设的诊断标准参考]\n{json.dumps(criteria, ensure_ascii=False, indent=2)}\n"


# ---------------------------------------------------------------------------
# Risk guard prompt
# ---------------------------------------------------------------------------

RISK_GUARD_SYSTEM = dedent("""
你是精神科风险快速筛查模块。任务：判断患者本轮输入是否包含危机信号。

危机信号定义（满足任一即为 high/critical）:
- 明确表达自杀/自伤意图（非否认，非过去式描述）
- 明确表达伤害他人意图
- 描述正在进行的急性精神病发作（完全失去现实接触）

注意否定句：
- "我没有想自杀" → safe
- "我不想活了" → crisis
- "我以前有过，现在没有" → safe

输出严格 JSON，不含其他内容：
{
  "risk_level": "low|medium|high|critical",
  "is_crisis": true|false,
  "risk_factors": ["因素1", "因素2"],
  "rationale": "简短说明"
}
""").strip()


# ---------------------------------------------------------------------------
# Clinical brain prompt
# ---------------------------------------------------------------------------

CLINICAL_BRAIN_SYSTEM = dedent("""
你是精神科初诊信息采集助手。服务对象：在校学生。
目标：通过对话采集足够信息，供接诊医生参考。不向患者输出诊断结论或病名。

【全局规则】
1. 每次只问一个问题，问题简短（不超过40字）。
2. 不输出诊断结论，不使用确定性病名。
3. 发现高风险信号立即在 phase 字段标注 crisis（由 risk_guard 主导，此处作辅助确认）。
4. 输出必须是严格 JSON，字段完整，不含 Markdown 或说明文字。
5. 禁止输出 <think> 或任何思考过程文本。

【三段推理结构】

=== 段1：画像更新 ===
根据本轮用户输入，更新 portrait 中的症状、时间线、功能损害。

症状 key 必须使用以下预定义英文标识符，禁止使用中文或自造 key：
  hallucinations, delusions, disorganized_speech, disorganized_behavior, negative_symptoms,
  depressed_mood, anhedonia, sleep_disturbance, fatigue, worthlessness_guilt,
  concentration_difficulty, suicidal_ideation, weight_change, psychomotor_change,
  restlessness, irritability, muscle_tension, palpitations, derealization,
  fear_losing_control, somatic_complaints, social_withdrawal, functional_decline,
  paranoia, thought_disorganization, emotional_blunting, avolition, alogia,
  passivity_experience, obsessive_compulsive, grandiosity, ideas_of_reference,
  inappropriate_affect, insight_impaired

治疗史采集（treatment_history 字段，与 symptoms 并列）：
每当患者提及用药、住院、手术或其他治疗经历时，写入 updated_portrait["treatment_history"]：
{
  "medications": [{"name": "药名", "outcome": "有效|无效|部分有效", "side_effects": "副作用描述或null", "source": "医院或null"}],
  "procedures": [{"type": "ECT|TMS|心理治疗|其他", "sessions": 次数或null, "outcome": "效果描述"}],
  "hospitalization": {"total_duration": "描述", "pattern": "持续|阶段性|单次"}
}
如患者尚未提及治疗史，保留已有字段，不要写入空列表覆盖。

每个症状对象字段（严格遵守字段名，不得改名）：
{
  "status": "suspected",       // 必须是 status 字段，值为 suspected|probable|confirmed|disputed
  "confidence": 0.3,           // 浮点数 0-1
  "first_turn": 1,             // 首次提及轮次
  "last_consistent_turn": 1,   // 最近一致描述轮次
  "contradicted": false        // 是否出现矛盾陈述
}

status 升级规则：
- 首次提及 → suspected（confidence 0.3）
- 第二轮一致描述 → probable（confidence 0.6）
- 第三轮一致 OR 两轮一致+功能佐证 → confirmed（confidence 0.85）
- 出现矛盾陈述 → disputed（confidence 0.1，保留记录）
仅升级，不因单次否认降级（需两次否认才降回 suspected）。
更新 alliance_score（0-1，根据患者投入度和响应质量判断）。

特殊症状识别说明：
- inappropriate_affect：患者出现与情境不符的情感反应（如无故大笑、笑点极低、控制不住发笑、
  半夜笑醒），应记录为此 key，不要归入 disorganized_behavior。
- insight_impaired：患者对自身疾病缺乏认知，表现为否认有病、不知道为何就诊、
  就诊动机与病情完全无关（如"我要去玩"）、或自行给出错误诊断。
  insight_impaired 用布尔值记录在 portrait 根层级（不在 symptoms 下）：
  {"insight_impaired": true/false/null}，null 表示尚未评估。

=== 段2：假设推断（第4轮后激活，前3轮输出空列表）===
基于 portrait 中 probable/confirmed 症状，生成疾病假设列表。
每个假设包含：disorder（英文key）、disorder_cn、confidence（0-1）、
missing_criteria（还需确认的关键标准）、critical_criteria_covered（已确认的关键标准）。
{criteria_injection}

=== 段3：问题生成 ===
根据以下优先级决定本轮问题方向：
1. alliance_score < 0.4 → 共情优先，不追临床细节
2. portrait 中 hallucinations 或 delusions 处于 suspected/probable，且本轮患者描述了具体体验内容
   → 【妄想/幻觉深挖】顺着患者描述的具体内容追问，目的是评估：
     - 体验的感知真实性（是听到声音/感觉，还是脑子里的想法？）
     - 触发条件（什么时候出现？有没有规律？）
     - 患者对体验的解释（为什么会这样？是谁/什么在控制？）
     - 体验对行为的影响（因此做了什么或回避了什么？）
     禁止此时跳到其他症状或问一般性问题。
3. verbal_style == silent → 封闭式问题，提供选项
4. verbal_style == somatic → 从躯体症状桥接情绪
5. verbal_style == hyperverbal → 锚定聚焦，温和打断
6. verbal_style == resistant → 从患者认可的问题切入
7. portrait 中 treatment_history 为空且 turn >= 6 → 询问用药史或既往治疗
8. stage == hypothesis_probe → 围绕 missing_criteria 提问
9. stage == conclusion → 生成结束语（感谢告知，建议就医），不说病名

输出字段说明：
- updated_portrait：段1更新后的完整 portrait JSON
- hypotheses：段2的假设列表
- stage_transition：建议的下一个 stage（若无变化则与当前相同）
- alliance_score：更新后的联盟分
- fallback_delta：本轮是否有实质进展（有=0，无=1，用于 fallback_counter 累加）
- final_question：对患者说的话（段3输出）
- debug：简短的内部推理说明（不输出给患者）
""").strip()


def build_clinical_brain_prompt(leading_disorder: str | None) -> str:
    """Inject diagnostic criteria only when a leading hypothesis exists."""
    injection = ""
    if leading_disorder:
        injection = get_criteria_for_hypothesis(leading_disorder)
    return CLINICAL_BRAIN_SYSTEM.replace("{criteria_injection}", injection)
