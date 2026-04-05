"""生成系统架构图，保存为 architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams
rcParams['font.family'] = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(20, 26))
ax.set_xlim(0, 20)
ax.set_ylim(0, 26)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")

# ── color palette ──────────────────────────────────────────────────────────
C_INPUT   = "#4A90D9"   # blue  - input layer
C_INIT    = "#7B68EE"   # purple - session init
C_PARALLEL= "#E8A838"   # orange - parallel analysis
C_RISK    = "#E05A5A"   # red   - risk
C_BRAIN   = "#3AAA6B"   # green - clinical brain
C_STATE   = "#5B9BD5"   # steel blue - state machine
C_DB      = "#8B6914"   # brown - database
C_OUTPUT  = "#888888"   # gray  - output
C_REPORT  = "#C44BC4"   # magenta - report
C_CRISIS  = "#CC0000"   # dark red - crisis
C_WHITE   = "#FFFFFF"
C_DARK    = "#2C2C2C"

def box(ax, x, y, w, h, label, sublabel="", color=C_WHITE, fontsize=11, textcolor=C_DARK, radius=0.3, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor=C_DARK, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    cy = y + h / 2 + (0.18 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.28, sublabel, ha="center", va="center",
                fontsize=8.5, color=textcolor, alpha=0.85, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_DARK, label="", lw=1.8, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.12, my, label, fontsize=8, color=color, zorder=5)

def side_arrow(ax, x1, y1, x2, y2, color=C_DARK, label="", rad=0.15):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6,
                                connectionstyle=f"arc3,rad={rad}"), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=8, color=color, zorder=5)

def dashed_box(ax, x, y, w, h, label, color="#DDDDDD"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.25",
                          facecolor=color, edgecolor="#AAAAAA",
                          linewidth=1.2, linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(x + 0.15, y + h - 0.28, label, fontsize=8.5, color="#666666",
            fontstyle="italic", zorder=2)

# ══════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════
ax.text(10, 25.5, "心理健康初诊系统 — 架构图", ha="center", va="center",
        fontsize=17, fontweight="bold", color=C_DARK,
        fontproperties="SimHei" if True else None)
ax.text(10, 25.05, "基于 LangGraph + DeepSeek + MySQL", ha="center", va="center",
        fontsize=10, color="#666666")

# ══════════════════════════════════════════════════════════════════════════
# Layer 0 — Patient input
# ══════════════════════════════════════════════════════════════════════════
box(ax, 7.5, 23.8, 5, 0.9, "患者输入", color=C_INPUT, textcolor=C_WHITE,
    fontsize=12, bold=True)

arrow(ax, 10, 23.8, 10, 23.2)

# ══════════════════════════════════════════════════════════════════════════
# Layer 1 — session_init
# ══════════════════════════════════════════════════════════════════════════
box(ax, 6, 22.2, 8, 0.95, "session_init", "学号采集 | 加载/创建 DB 记录",
    color=C_INIT, textcolor=C_WHITE, fontsize=11, bold=True)

# branch: awaiting student id
ax.annotate("", xy=(16.5, 22.65), xytext=(14, 22.65),
            arrowprops=dict(arrowstyle="->", color=C_OUTPUT, lw=1.5,
                            connectionstyle="arc3,rad=0.0"), zorder=2)
box(ax, 16.5, 22.2, 3, 0.9, "等待学号", "→ output_safety → END",
    color="#CCCCCC", fontsize=9)

arrow(ax, 10, 22.2, 10, 21.55, label="  学号已知")

# ══════════════════════════════════════════════════════════════════════════
# Layer 2 — input_safety
# ══════════════════════════════════════════════════════════════════════════
box(ax, 7, 21.4, 6, 0.9, "input_safety", "空输入拦截 | 追加 history",
    color="#AAAAAA", textcolor=C_WHITE, fontsize=10)

arrow(ax, 10, 21.4, 10, 20.7)

# ══════════════════════════════════════════════════════════════════════════
# Layer 3 — Parallel analysis
# ══════════════════════════════════════════════════════════════════════════
dashed_box(ax, 1.2, 18.7, 17.6, 1.85, "analyze node（并行执行，目标 <10s）", "#FFF8E8")

# risk_guard
box(ax, 1.8, 19.0, 6.5, 1.2,
    "risk_guard", "DeepSeek-chat | max_tokens=300\n关键词+LLM双重判断",
    color=C_RISK, textcolor=C_WHITE, fontsize=10)

# clinical_brain
box(ax, 10.5, 19.0, 7.8, 1.2,
    "clinical_brain", "DeepSeek-chat | max_tokens=1800\n画像更新 | 假设推断 | 问题生成",
    color=C_BRAIN, textcolor=C_WHITE, fontsize=10)

# fan-out arrows
ax.annotate("", xy=(5.05, 19.6), xytext=(10, 20.7),
            arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.5,
                            connectionstyle="arc3,rad=0.15"), zorder=2)
ax.annotate("", xy=(14.39, 19.6), xytext=(10, 20.7),
            arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.5,
                            connectionstyle="arc3,rad=-0.15"), zorder=2)
ax.text(5.5, 20.35, "asyncio.gather", fontsize=8.5, color="#888888", style="italic")

arrow(ax, 5.05, 19.0, 9.2, 18.1, color=C_RISK)
arrow(ax, 14.39, 19.0, 10.8, 18.1, color=C_BRAIN)

# ══════════════════════════════════════════════════════════════════════════
# Layer 4 — route_by_risk
# ══════════════════════════════════════════════════════════════════════════
box(ax, 7.2, 17.35, 5.6, 0.75, "route_by_risk()", color="#E8E8E8", fontsize=10)

# crisis branch
ax.annotate("", xy=(3.2, 16.5), xytext=(7.2, 17.72),
            arrowprops=dict(arrowstyle="->", color=C_CRISIS, lw=2.0,
                            connectionstyle="arc3,rad=0.25"), zorder=2)
ax.text(3.8, 17.3, "is_crisis=True", fontsize=8.5, color=C_CRISIS, fontweight="bold")

box(ax, 0.5, 15.6, 5.5, 0.9, "crisis_node", "固定安全响应 | 急救资源",
    color=C_CRISIS, textcolor=C_WHITE, fontsize=10)

arrow(ax, 10, 17.35, 10, 16.7, label="  safe")

# ══════════════════════════════════════════════════════════════════════════
# Layer 5 — State machine (内嵌展示)
# ══════════════════════════════════════════════════════════════════════════
dashed_box(ax, 1.0, 13.5, 18, 3.0, "状态机（clinical_brain 内部驱动）", "#EEF4FF")

# stage boxes
stages = [
    (1.4,  14.8, "ENTRY\nDETECTION", C_STATE),
    (4.5,  15.5, "ACTIVE\nLISTEN",   C_BRAIN),
    (4.5,  14.2, "STRUCTURED\nPROBE",C_BRAIN),
    (7.8,  15.5, "RAPPORT\nBUILD",   "#E8A838"),
    (7.8,  14.2, "SOMATIC\nBRIDGE",  "#E8A838"),
    (7.8,  12.95,"ANCHORING",        "#E8A838"),
    (11.2, 14.5, "HYPOTHESIS\nPROBE",C_INPUT),
    (14.8, 14.5, "CONCLUSION",       C_REPORT),
    (14.8, 13.2, "FORCED\nCLOSURE",  "#888888"),
]
for sx, sy, slabel, sc in stages:
    box(ax, sx, sy, 2.8, 0.88, slabel, color=sc, textcolor=C_WHITE, fontsize=8, radius=0.2)

# state transitions (simplified)
# entry → active_listen
ax.annotate("", xy=(4.5, 15.94), xytext=(4.2, 15.5),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2), zorder=2)
ax.annotate("", xy=(4.5, 14.64), xytext=(4.2, 14.64),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2), zorder=2)
# active/structured → rapport
ax.annotate("", xy=(7.8, 15.94), xytext=(7.3, 15.94),
            arrowprops=dict(arrowstyle="->", color=C_RISK, lw=1.1,
                            connectionstyle="arc3,rad=0.0"), zorder=2)
# → hypothesis probe
ax.annotate("", xy=(11.2, 14.94), xytext=(10.6, 14.94),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2), zorder=2)
# → conclusion
ax.annotate("", xy=(14.8, 14.94), xytext=(14.0, 14.94),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2), zorder=2)

ax.text(1.5, 13.65, "症状置信度: suspected → probable → confirmed → disputed", fontsize=8, color="#555")

arrow(ax, 10, 16.7, 10, 16.5)
ax.annotate("", xy=(10, 13.5), xytext=(10, 16.5),
            arrowprops=dict(arrowstyle="->", color=C_STATE, lw=1.5,
                            connectionstyle="arc3,rad=0.0"), zorder=2)

arrow(ax, 10, 13.5, 10, 12.85)

# ══════════════════════════════════════════════════════════════════════════
# Layer 6 — output_safety
# ══════════════════════════════════════════════════════════════════════════
box(ax, 7, 12.1, 6, 0.75, "output_safety", "过滤病名 | 写入 history",
    color="#AAAAAA", textcolor=C_WHITE, fontsize=10)

# crisis → output_safety
ax.annotate("", xy=(7.0, 12.47), xytext=(6.0, 16.05),
            arrowprops=dict(arrowstyle="->", color=C_CRISIS, lw=1.5,
                            connectionstyle="arc3,rad=-0.25"), zorder=2)

arrow(ax, 10, 12.1, 10, 11.45, label="  conclusion/crisis/forced_closure")

# ══════════════════════════════════════════════════════════════════════════
# Layer 7 — generate_report
# ══════════════════════════════════════════════════════════════════════════
box(ax, 6.5, 10.55, 7, 0.9, "generate_report",
    "DeepSeek 生成临床摘要（含病名，仅给医生）",
    color=C_REPORT, textcolor=C_WHITE, fontsize=10)

arrow(ax, 10, 10.55, 10, 9.85)

# ══════════════════════════════════════════════════════════════════════════
# Layer 8 — DB
# ══════════════════════════════════════════════════════════════════════════
dashed_box(ax, 1.2, 8.0, 17.6, 1.7, "MySQL 持久化层", "#FFF3E0")

db_tables = [
    (1.8,  8.25, "patients\n学号 | verbal_style\n就诊次数"),
    (6.5,  8.25, "clinical_portraits\n症状画像 JSON\nprobable+ 才持久化"),
    (11.5, 8.25, "session_records\n每次会话记录\n报告 JSON"),
]
for dx, dy, dlabel in db_tables:
    box(ax, dx, dy, 4.2, 1.3, dlabel, color=C_DB, textcolor=C_WHITE, fontsize=8.5, radius=0.2)

# arrows to DB
ax.annotate("", xy=(4.0, 9.55), xytext=(9.0, 9.85),
            arrowprops=dict(arrowstyle="->", color=C_DB, lw=1.3,
                            connectionstyle="arc3,rad=0.2"), zorder=2)
ax.annotate("", xy=(8.6, 9.55), xytext=(10.0, 9.85),
            arrowprops=dict(arrowstyle="->", color=C_DB, lw=1.3,
                            connectionstyle="arc3,rad=0.05"), zorder=2)
ax.annotate("", xy=(13.6, 9.55), xytext=(11.0, 9.85),
            arrowprops=dict(arrowstyle="->", color=C_DB, lw=1.3,
                            connectionstyle="arc3,rad=-0.2"), zorder=2)

# save_portrait every turn
ax.annotate("", xy=(8.6, 9.55), xytext=(10.0, 12.1),
            arrowprops=dict(arrowstyle="->", color=C_DB, lw=1.1, linestyle="dashed",
                            connectionstyle="arc3,rad=-0.3"), zorder=2)
ax.text(11.8, 10.85, "每轮更新画像", fontsize=7.5, color=C_DB, style="italic")

# load on init
ax.annotate("", xy=(8.6, 9.55), xytext=(7.0, 22.2),
            arrowprops=dict(arrowstyle="<-", color=C_DB, lw=1.1, linestyle="dashed",
                            connectionstyle="arc3,rad=0.35"), zorder=2)
ax.text(0.3, 16.0, "加载历史\n画像", fontsize=7.5, color=C_DB, style="italic", ha="center")

arrow(ax, 10, 8.0, 10, 7.3)

# ══════════════════════════════════════════════════════════════════════════
# Layer 9 — Output to patient / doctor
# ══════════════════════════════════════════════════════════════════════════
dashed_box(ax, 2.0, 5.8, 16, 1.4, "输出层", "#F0FFF0")

box(ax, 2.5, 6.0, 6.5, 1.0, "对患者的回复",
    "追问 | 共情 | 建议就医\n（不含病名）",
    color=C_BRAIN, textcolor=C_WHITE, fontsize=10)

box(ax, 11.0, 6.0, 6.5, 1.0, "临床报告（医生专用）",
    "病名 + 置信度 + 症状证据链\n建议下一步",
    color=C_REPORT, textcolor=C_WHITE, fontsize=10)

ax.annotate("", xy=(5.75, 7.0), xytext=(10.0, 7.3),
            arrowprops=dict(arrowstyle="->", color=C_BRAIN, lw=1.5,
                            connectionstyle="arc3,rad=0.2"), zorder=2)
ax.annotate("", xy=(14.25, 7.0), xytext=(10.0, 7.3),
            arrowprops=dict(arrowstyle="->", color=C_REPORT, lw=1.5,
                            connectionstyle="arc3,rad=-0.2"), zorder=2)

# ══════════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_INPUT,   "输入 / 路由层"),
    (C_INIT,    "会话初始化"),
    (C_RISK,    "风险检测"),
    (C_BRAIN,   "临床推理"),
    (C_STATE,   "状态机"),
    (C_DB,      "数据库"),
    (C_REPORT,  "报告 / 结论"),
    (C_CRISIS,  "危机处置"),
]
lx, ly = 1.0, 5.0
for i, (lc, lt) in enumerate(legend_items):
    col = i % 4
    row = i // 4
    bx = lx + col * 4.5
    by = ly - row * 0.6
    rect = FancyBboxPatch((bx, by-0.18), 0.5, 0.36,
                          boxstyle="round,pad=0.02",
                          facecolor=lc, edgecolor="none", zorder=3)
    ax.add_patch(rect)
    ax.text(bx + 0.65, by, lt, va="center", fontsize=9, color=C_DARK, zorder=4)

ax.text(10, 4.0, "文件: src/psy_debate/ │ DB: MySQL psy_debate │ LLM: DeepSeek-chat (cloud)",
        ha="center", va="center", fontsize=8.5, color="#888888")

plt.tight_layout(pad=0.5)
out = "C:/Users/31906/Desktop/1/architecture.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print("Saved:", out)
