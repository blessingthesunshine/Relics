import tkinter as tk
from tkinter import ttk, messagebox
from math import comb
from functools import lru_cache
import numpy as np


# ============================================================
# Genshin artifact probability calculator
# Model:
# - Each drop independently goes to one of 5 slots with probability 1/5.
# - Four visible substats are sampled uniformly without replacement.
# - Initial 3-line artifact: 4 upgrades, probability p3.
# - Initial 4-line artifact: 5 upgrades, probability 1-p3.
# - Each upgrade independently chooses one of the 4 substats uniformly.
# - "a" and "b" are minimum initial effective-substat counts for screening.
# - For each slot, keep the best qualifying artifact among N drops.
# ============================================================

SANDS = {
    "生命百分比": (4 / 15, 9, 2),
    "攻击百分比": (4 / 15, 9, 2),
    "防御百分比": (4 / 15, 9, 2),
    "元素充能效率": (0.10, 9, 2),
    "元素精通": (0.10, 9, 2),
}

CUP = {
    "生命百分比": (0.19175, 9, 2),
    "攻击百分比": (0.19175, 9, 2),
    "防御百分比": (0.19150, 9, 2),
    "火元素伤害加成": (0.05, 10, 3),
    "水元素伤害加成": (0.05, 10, 3),
    "风元素伤害加成": (0.05, 10, 3),
    "雷元素伤害加成": (0.05, 10, 3),
    "草元素伤害加成": (0.05, 10, 3),
    "冰元素伤害加成": (0.05, 10, 3),
    "岩元素伤害加成": (0.05, 10, 3),
    "物理伤害加成": (0.05, 10, 3),
    "元素精通": (0.025, 9, 2),
}

CROWN = {
    "生命百分比": (0.22, 9, 2),
    "攻击百分比": (0.22, 9, 2),
    "防御百分比": (0.22, 9, 2),
    "暴击率": (0.10, 9, 2),
    "暴击伤害": (0.10, 9, 2),
    "治疗加成": (0.10, 10, 3),
    "元素精通": (0.04, 9, 2),
}


def binomial_pmf(n, p):
    return [
        comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        for k in range(n + 1)
    ]


def one_slot_mass(M, K, min_initial_effective, main_probability, p3):
    """
    mass[t] = P(one random artifact is in this slot, has the selected
               correct main stat, passes initial screening, and ends
               with exactly t effective substat rolls).
    t ranges from 0 to 8.
    """
    mass = np.zeros(9, dtype=np.longdouble)
    denominator = comb(M, 4)

    for j in range(0, min(4, K) + 1):
        if j < min_initial_effective:
            continue
        if 4 - j > M - K:
            continue

        p_initial = (
            comb(K, j) * comb(M - K, 4 - j) / denominator
        )

        # Initial 3-line artifact: 4 upgrades.
        # Initial 4-line artifact: 5 upgrades.
        for upgrades, probability in ((4, p3), (5, 1 - p3)):
            if probability == 0:
                continue
            for added, p_added in enumerate(binomial_pmf(upgrades, j / 4)):
                final_count = j + added
                mass[final_count] += (
                    np.longdouble(0.2)
                    * np.longdouble(main_probability)
                    * np.longdouble(p_initial)
                    * np.longdouble(probability)
                    * np.longdouble(p_added)
                )

    return mass


@lru_cache(maxsize=64)
def total_distribution(N, a, b, p3,
                        sands_choice, cup_choice, crown_choice):
    """
    Exact finite-state calculation for:
    P(sum of the five slot-wise best qualifying artifact counts > v).
    The maximum for a slot is -1 if that slot has no qualifying artifact.
    """
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError("N必须是非负整数。")
    if not 0 <= a <= 3 or not 0 <= b <= 3 or not 0 <= p3 <= 1:
        raise ValueError("筛选下限须在0到3之间，p3须在0到1之间。")
    s_prob, s_M, s_K = SANDS[sands_choice]
    c_prob, c_M, c_K = CUP[cup_choice]
    r_prob, r_M, r_K = CROWN[crown_choice]

    masses = [
        one_slot_mass(9, 3, a, 1.0, p3),       # Flower
        one_slot_mass(9, 3, a, 1.0, p3),       # Feather
        one_slot_mass(s_M, s_K, b, s_prob, p3),  # Sands
        one_slot_mass(c_M, c_K, b, c_prob, p3),  # Goblet
        one_slot_mass(r_M, r_K, b, r_prob, p3),  # Circlet
    ]

    # For each slot, tail[i, x] = P(qualifying artifact has T > x).
    # x = -1, 0, ..., 8. A maximum value of -1 means no qualifying item.
    x_values = np.arange(-1, 9)
    tails = []
    for mass in masses:
        tail = np.zeros(10, dtype=np.longdouble)
        for idx, x in enumerate(x_values):
            tail[idx] = np.sum(mass[x + 1:]) if x < 8 else 0
        tails.append(tail)

    # F(x1,...,x5) = P(M1<=x1,...,M5<=x5)
    #             = (1 - sum_i P(one drop is qualifying in slot i with T>xi))^N
    total_tail = np.zeros((10, 10, 10, 10, 10), dtype=np.longdouble)
    for axis, tail in enumerate(tails):
        shape = [1] * 5
        shape[axis] = 10
        total_tail += tail.reshape(shape)

    base = np.clip(1 - total_tail, 0, 1)
    cdf = np.power(base, N)

    # Five-dimensional finite differences convert the joint CDF to PMF.
    pmf = cdf
    for axis in range(5):
        first_slice = np.take(pmf, [0], axis=axis)
        pmf = np.diff(pmf, axis=axis, prepend=np.zeros_like(first_slice))

    # Check the raw finite differences before correcting floating-point noise.
    if np.min(pmf) < -1e-10 or abs(float(pmf.sum()) - 1) > 1e-10:
        raise ArithmeticError("联合分布数值校验失败。")
    pmf = np.maximum(pmf, 0)
    # A successful set must contain all five slots, even at low thresholds.
    valid = pmf[1:, 1:, 1:, 1:, 1:]
    totals = np.indices(valid.shape).sum(axis=0)
    distribution = np.bincount(totals.ravel(), weights=valid.astype(float).ravel(), minlength=41)
    distribution.setflags(write=False)
    return distribution


def probability_for_N_v(N, v, a, b, p3, sands_choice, cup_choice, crown_choice):
    """Probability of a complete five-piece set with total strictly above v."""
    distribution = total_distribution(N, a, b, p3, sands_choice, cup_choice, crown_choice)
    return min(1.0, float(distribution[np.arange(41) > v].sum()))


def solve_v(N, target, args):
    """Largest integer v for which P(total > v) >= target."""
    if target == 0:
        return 40, probability_for_N_v(N, 40, *args)
    candidates = []
    for v in range(-1, 41):
        p = probability_for_N_v(N, v, *args)
        if p >= target:
            candidates.append((v, p))
    if not candidates:
        return None, None
    return candidates[-1]


def solve_N(v, target, args):
    """Smallest integer N for which P(total > v) >= target."""
    if target == 0:
        return 0, probability_for_N_v(0, v, *args)

    if probability_for_N_v(0, v, *args) >= target:
        return 0, probability_for_N_v(0, v, *args)
    if v >= 40 or args[1] == 3:
        # Sands have at most two effective initial substats.
        return None, None
    lo, hi = 0, 1
    while hi < 10_000_000 and probability_for_N_v(hi, v, *args) < target:
        lo = hi
        hi = min(10_000_000, hi * 2)

    if hi >= 10_000_000 and probability_for_N_v(hi, v, *args) < target:
        return None, None

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if probability_for_N_v(mid, v, *args) >= target:
            hi = mid
        else:
            lo = mid

    return hi, probability_for_N_v(hi, v, *args)


class ArtifactCalculator:
    def __init__(self, root):
        self.root = root
        root.title("原神圣遗物刷取概率计算器")
        root.geometry("760x760")
        root.minsize(700, 680)

        main = ttk.Frame(root, padding=14)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main, text="原神圣遗物刷取概率计算器",
            font=("Microsoft YaHei UI", 16, "bold")
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            main,
            text="a、b为初始有效副词条筛选下限；p3为初始3词条概率。",
            wraplength=700
        ).pack(anchor="w", pady=(0, 10))

        settings = ttk.LabelFrame(main, text="筛选与主词条设置", padding=10)
        settings.pack(fill="x", pady=5)

        self.a_var = tk.StringVar(value="3")
        self.b_var = tk.StringVar(value="2")
        self.p3_var = tk.StringVar(value="0.8")

        row = ttk.Frame(settings)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="花、羽初始有效词条下限 a：").pack(side="left")
        ttk.Entry(row, textvariable=self.a_var, width=8).pack(side="left", padx=5)
        ttk.Label(row, text="沙、杯、冠初始有效词条下限 b：").pack(side="left", padx=(15, 0))
        ttk.Entry(row, textvariable=self.b_var, width=8).pack(side="left", padx=5)
        ttk.Label(row, text="初始3词条概率 p3：").pack(side="left", padx=(15, 0))
        ttk.Entry(row, textvariable=self.p3_var, width=8).pack(side="left", padx=5)

        self.sands_var = tk.StringVar(value="生命百分比")
        self.cup_var = tk.StringVar(value="火元素伤害加成")
        self.crown_var = tk.StringVar(value="暴击率")

        for label, var, options in [
            ("沙漏正确主词条：", self.sands_var, list(SANDS.keys())),
            ("杯子正确主词条：", self.cup_var, list(CUP.keys())),
            ("头冠正确主词条：", self.crown_var, list(CROWN.keys())),
        ]:
            r = ttk.Frame(settings)
            r.pack(fill="x", pady=4)
            ttk.Label(r, text=label, width=22).pack(side="left")
            ttk.Combobox(
                r, textvariable=var, values=options,
                state="readonly", width=28
            ).pack(side="left")

        ttk.Label(
            settings,
            text="说明：花、羽主词条固定正确；沙/杯/冠按所选主词条概率计算。",
            foreground="#555"
        ).pack(anchor="w", pady=(5, 0))

        calcbox = ttk.LabelFrame(
            main, text="输入 N、v、f 中任意两个", padding=10
        )
        calcbox.pack(fill="x", pady=8)

        self.N_var = tk.StringVar(value="200")
        self.v_var = tk.StringVar(value="32")
        self.f_var = tk.StringVar(value="")

        for label, var, hint in [
            ("刷取总数 N：", self.N_var, "非负整数"),
            ("总有效词条阈值 v：", self.v_var, "要求总数严格大于 v"),
            ("目标概率 f：", self.f_var, "小数，例如 0.5"),
        ]:
            r = ttk.Frame(calcbox)
            r.pack(fill="x", pady=4)
            ttk.Label(r, text=label, width=22).pack(side="left")
            ttk.Entry(r, textvariable=var, width=16).pack(side="left")
            ttk.Label(r, text=hint, foreground="#666").pack(side="left", padx=8)

        ttk.Button(
            calcbox, text="计算", command=self.calculate
        ).pack(anchor="e", pady=(8, 0))

        resultbox = ttk.LabelFrame(main, text="计算结果", padding=10)
        resultbox.pack(fill="both", expand=True, pady=5)

        self.output = tk.Text(
            resultbox, height=16, wrap="word",
            font=("Microsoft YaHei UI", 10)
        )
        self.output.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="模型假设：每次强化独立地以1/4概率命中四个副词条之一；"
                 "初始3词条强化4次，初始4词条强化5次。",
            foreground="#555", wraplength=700
        ).pack(anchor="w", pady=(5, 0))

    def get_settings(self):
        a = int(self.a_var.get())
        b = int(self.b_var.get())
        p3 = float(self.p3_var.get())

        if not 0 <= a <= 3:
            raise ValueError("a必须是0到3之间的整数。")
        if not 0 <= b <= 3:
            raise ValueError("b必须是0到3之间的整数。")
        if not 0 <= p3 <= 1:
            raise ValueError("p3必须在0到1之间。")

        return (a, b, p3,
                self.sands_var.get(),
                self.cup_var.get(),
                self.crown_var.get())

    def calculate(self):
        try:
            args = self.get_settings()
            raw = [self.N_var.get().strip(),
                   self.v_var.get().strip(),
                   self.f_var.get().strip()]
            if sum(bool(x) for x in raw) != 2:
                raise ValueError("N、v、f必须恰好填写两个，另一个留空。")

            self.output.delete("1.0", "end")

            if raw[0] and raw[1]:
                N = int(raw[0])
                v = int(raw[1])
                if N < 0:
                    raise ValueError("N必须大于等于0。")
                p = probability_for_N_v(N, v, *args)
                self.f_var.set(f"{p:.12g}")
                self.output.insert(
                    "end",
                    f"已知 N={N}, v={v}\n"
                    f"所求成功概率 f = {p:.12%}\n"
                    f"小数形式：{p:.12g}\n"
                )

            elif raw[0] and raw[2]:
                N = int(raw[0])
                target = float(raw[2])
                if N < 0:
                    raise ValueError("N必须大于等于0。")
                if not 0 <= target < 1:
                    raise ValueError("f必须满足 0 <= f < 1。")
                v, p = solve_v(N, target, args)
                if v is None:
                    self.output.insert(
                        "end", "在可取阈值范围内，没有找到满足目标概率的 v。"
                    )
                else:
                    self.v_var.set(str(v))
                    self.output.insert(
                        "end",
                        f"已知 N={N}, 目标概率 f={target:.12g}\n"
                        f"所求最大整数阈值 v={v}\n"
                        f"P(总有效词条数 > {v})={p:.12%}\n"
                        f"该 v 是满足概率不低于目标值的最大整数阈值。\n"
                    )

            else:
                v = int(raw[1])
                target = float(raw[2])
                if not 0 <= target < 1:
                    raise ValueError("f必须满足 0 <= f < 1。")
                N, p = solve_N(v, target, args)
                if N is None:
                    self.output.insert(
                        "end", "目标不可达，或所需 N 超出程序搜索上限 10,000,000。"
                    )
                else:
                    self.N_var.set(str(N))
                    self.output.insert(
                        "end",
                        f"已知 v={v}, 目标概率 f={target:.12g}\n"
                        f"所需最小刷取数量 N={N}\n"
                        f"P(总有效词条数 > {v})={p:.12%}\n"
                    )

            self.output.insert(
                "end",
                "\n当前主词条选择：\n"
                f"沙：{self.sands_var.get()}\n"
                f"杯：{self.cup_var.get()}\n"
                f"冠：{self.crown_var.get()}\n"
                f"a={args[0]}, b={args[1]}, p3={args[2]}\n"
            )

        except Exception as e:
            messagebox.showerror("输入错误", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ArtifactCalculator(root)
    root.mainloop()
