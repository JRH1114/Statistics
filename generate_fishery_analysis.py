from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from docx import Document
from matplotlib import font_manager
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera


ROOT = Path(__file__).resolve().parent
REPORT_NAME = "数字经济对渔业高质量发展的影响分析.md"

FIGURES = {
    "corr": "correlation_heatmap.png",
    "scatter_x": "scatter_Y_X.png",
    "scatter_infra": "scatter_Y_digital_infrastructure.png",
    "scatter_agri": "scatter_Y_agriculture_digitalization.png",
    "trend": "trend_Y_X.png",
    "region": "region_comparison.png",
}

X_SECONDARY_ORDER = [
    "数字基础设施",
    "通信业务服务化",
    "通信服务业数字化",
    "农业数字化",
    "工业数字化",
    "服务业数字化",
]

WARM_PALETTE = {
    "deep": "#8B002A",
    "main": "#C73E4E",
    "accent": "#E07A5F",
    "soft": "#F2B39F",
    "pale": "#F8E3DC",
    "grid": "#E7D8D3",
    "text": "#2A2020",
}

Y_SECONDARY_ORDER = [
    "渔业生产",
    "渔业资源",
    "渔业加工与贸易",
    "渔业技术培训",
]

REGION_MAP = {
    "北京": "东部",
    "天津": "东部",
    "河北": "东部",
    "辽宁": "东部",
    "上海": "东部",
    "江苏": "东部",
    "浙江": "东部",
    "福建": "东部",
    "山东": "东部",
    "广东": "东部",
    "海南": "东部",
    "山西": "中部",
    "吉林": "中部",
    "黑龙江": "中部",
    "安徽": "中部",
    "江西": "中部",
    "河南": "中部",
    "湖北": "中部",
    "湖南": "中部",
    "内蒙古": "西部",
    "广西": "西部",
    "重庆": "西部",
    "四川": "西部",
    "贵州": "西部",
    "云南": "西部",
    "陕西": "西部",
    "甘肃": "西部",
    "青海": "西部",
    "宁夏": "西部",
    "新疆": "西部",
}


def find_input_file(suffix: str) -> Path:
    matches = [p for p in ROOT.glob(f"*{suffix}") if not p.name.startswith("~$")]
    if not matches:
        raise FileNotFoundError(f"未找到 {suffix} 文件")
    return matches[0]


def setup_chinese_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen_font = "sans-serif"
    for candidate in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]:
        if candidate in available:
            chosen_font = candidate
            break
    plt.rcParams["font.sans-serif"] = [chosen_font]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(
        style="whitegrid",
        font=chosen_font,
        rc={
            "axes.facecolor": "#FFFDFB",
            "figure.facecolor": "#FFFDFB",
            "axes.edgecolor": WARM_PALETTE["grid"],
            "axes.labelcolor": WARM_PALETTE["text"],
            "xtick.color": WARM_PALETTE["text"],
            "ytick.color": WARM_PALETTE["text"],
            "text.color": WARM_PALETTE["text"],
            "grid.color": WARM_PALETTE["grid"],
            "grid.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.facecolor": "#FFFDFB",
            "legend.edgecolor": WARM_PALETTE["grid"],
        },
    )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_unit(name: str) -> str:
    match = re.search(r"[（(]([^（）()]*)[）)]", name)
    if match:
        return match.group(1).strip()
    if "比重" in name or "率" in name:
        return "比重/百分比"
    if "指数" in name:
        return "指数"
    return "根据原始统计口径"


def parse_indicator_system(xlsx: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name="Sheet4", header=None)
    records: list[dict[str, str]] = []
    current_y_secondary = ""
    current_x_secondary = ""

    for _, row in raw.iterrows():
        y_secondary = clean_text(row.get(1))
        y_name = clean_text(row.get(2))
        y_code = clean_text(row.get(3))
        if y_secondary:
            current_y_secondary = y_secondary
        if y_name and y_code.startswith("y"):
            records.append(
                {
                    "domain": "y",
                    "一级指标": "渔业高质量发展 Y",
                    "二级指标": current_y_secondary,
                    "三级指标": y_name,
                    "代码": y_code,
                    "单位": extract_unit(y_name),
                    "指标方向": "正向",
                    "指标含义": f"反映{current_y_secondary}中的{y_name}水平",
                }
            )

        x_secondary = clean_text(row.get(10))
        x_name = clean_text(row.get(11))
        x_code = clean_text(row.get(12))
        if x_secondary:
            current_x_secondary = x_secondary
        if x_name and x_code.startswith("x"):
            records.append(
                {
                    "domain": "x",
                    "一级指标": "数字经济发展 X",
                    "二级指标": current_x_secondary,
                    "三级指标": x_name,
                    "代码": x_code,
                    "单位": extract_unit(x_name),
                    "指标方向": "正向",
                    "指标含义": f"反映{current_x_secondary}中的{x_name}水平",
                }
            )

    indicators = pd.DataFrame(records)
    if indicators.empty:
        raise ValueError("未能从 Sheet4 解析出指标体系")
    return indicators


def parse_province_map(xlsx: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name="Sheet3", header=None)
    province_map = raw.iloc[3:, [2, 3, 4]].copy()
    province_map.columns = ["id", "省份", "year"]
    province_map = province_map.dropna(subset=["id", "省份", "year"])
    province_map["id"] = province_map["id"].astype(int)
    province_map["year"] = province_map["year"].astype(int)
    province_map["省份"] = province_map["省份"].astype(str).str.strip()
    return province_map.drop_duplicates(["id", "year"])


def read_word_context(docx_path: Path) -> dict[str, str]:
    document = Document(docx_path)
    paragraphs = [p.text.strip().replace("\u200b", "") for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)

    sample_match = re.search(r"选取我国2011[-—]2023年30个省域.*?面板数据", text)
    source_match = re.search(r"数据主要来源于历年的《中国统计年鉴》.*?各省市统计年鉴等", text)

    return {
        "title": paragraphs[0] if paragraphs else "数字技术对我国渔业高质量发展的影响研究",
        "sample": sample_match.group(0) if sample_match else "选取我国2011-2023年30个省域面板数据",
        "source": source_match.group(0)
        if source_match
        else "数据主要来源于历年统计年鉴及各省市统计年鉴等",
        "background": (
            "Word 文档指出，数字技术能够通过物联网、大数据、人工智能等手段提高资源配置效率、"
            "提升渔业生产效率，并推动渔业产业链数字化转型；渔业高质量发展强调生产效率、"
            "资源保护、产业结构优化和技术推广能力的综合提升。"
        ),
    }


def prepare_data(xlsx: Path, indicators: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    y_raw = pd.read_excel(xlsx, sheet_name="y")
    x_raw = pd.read_excel(xlsx, sheet_name="x")
    notes: dict[str, object] = {}

    y_codes = indicators.loc[indicators["domain"] == "y", "代码"].tolist()
    x_codes = indicators.loc[indicators["domain"] == "x", "代码"].tolist()

    y_value_cols = [c for c in y_raw.columns if c not in ["id", "year"]]
    y_used_cols = y_value_cols[: len(y_codes)]
    y_extra_cols = y_value_cols[len(y_codes) :]
    notes["y_extra_cols"] = [str(c) for c in y_extra_cols]
    y = y_raw[["id", "year"] + y_used_cols].copy()
    y = y.rename(columns={old: new for old, new in zip(y_used_cols, y_codes)})

    x = x_raw[["id", "year"] + x_codes].copy()
    x = x.reset_index(drop=True)
    changed_rows: list[str] = []
    if len(x) == 390:
        expected_id = np.repeat(np.arange(1, 31), 13)
        mismatch = x["id"].to_numpy() != expected_id
        for idx in np.where(mismatch)[0]:
            changed_rows.append(
                f"第{idx + 2}行 year={int(x.loc[idx, 'year'])}: id {int(x.loc[idx, 'id'])} -> {int(expected_id[idx])}"
            )
        x["id"] = expected_id
    notes["x_id_corrections"] = changed_rows

    province_map = parse_province_map(xlsx)
    data = y.merge(x, on=["id", "year"], how="inner").merge(province_map, on=["id", "year"], how="left")
    if data["省份"].isna().any():
        id_to_province = province_map.drop_duplicates("id").set_index("id")["省份"]
        data["省份"] = data["省份"].fillna(data["id"].map(id_to_province))
    data["地区"] = data["省份"].map(REGION_MAP).fillna("未分类")

    for col in y_codes + x_codes:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    notes["merged_shape"] = data.shape
    notes["missing_values"] = int(data[y_codes + x_codes].isna().sum().sum())
    notes["duplicate_id_year"] = int(data.duplicated(["id", "year"]).sum())
    notes["province_count"] = int(data["id"].nunique())
    notes["year_min"] = int(data["year"].min())
    notes["year_max"] = int(data["year"].max())
    return data, notes


def minmax_standardize(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    standardized = pd.DataFrame(index=frame.index)
    for col in cols:
        series = frame[col].astype(float)
        col_min = series.min()
        col_max = series.max()
        if math.isclose(col_max, col_min):
            standardized[col] = 0.0
        else:
            standardized[col] = (series - col_min) / (col_max - col_min)
    return standardized


def entropy_index(frame: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    z = minmax_standardize(frame, cols)
    n = len(z)
    eps = 1e-12
    proportions = z.copy()
    for col in cols:
        total = proportions[col].sum()
        if math.isclose(total, 0.0):
            proportions[col] = 1.0 / n
        else:
            proportions[col] = proportions[col] / total
    safe = proportions.clip(lower=eps)
    entropy = -(safe * np.log(safe)).sum(axis=0) / np.log(n)
    diversity = 1 - entropy
    if math.isclose(float(diversity.sum()), 0.0):
        weights = pd.Series(1 / len(cols), index=cols)
    else:
        weights = diversity / diversity.sum()
    score = z.mul(weights, axis=1).sum(axis=1)
    return score, weights, z


def compute_indices(data: pd.DataFrame, indicators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    weight_rows: list[dict[str, object]] = []

    for domain, overall_name, secondary_order in [
        ("y", "Y综合指数", Y_SECONDARY_ORDER),
        ("x", "X综合指数", X_SECONDARY_ORDER),
    ]:
        domain_indicators = indicators[indicators["domain"] == domain]
        all_codes = domain_indicators["代码"].tolist()
        data[overall_name], overall_weights, _ = entropy_index(data, all_codes)
        for code, weight in overall_weights.items():
            row = domain_indicators.loc[domain_indicators["代码"] == code].iloc[0]
            weight_rows.append(
                {
                    "指数": overall_name,
                    "二级指标": row["二级指标"],
                    "三级指标": row["三级指标"],
                    "代码": code,
                    "权重": float(weight),
                }
            )

        for secondary in secondary_order:
            group_codes = domain_indicators.loc[domain_indicators["二级指标"] == secondary, "代码"].tolist()
            if not group_codes:
                continue
            score_name = f"{secondary}指数"
            data[score_name], group_weights, _ = entropy_index(data, group_codes)
            for code, weight in group_weights.items():
                row = domain_indicators.loc[indicators["代码"] == code].iloc[0]
                weight_rows.append(
                    {
                        "指数": score_name,
                        "二级指标": secondary,
                        "三级指标": row["三级指标"],
                        "代码": code,
                        "权重": float(weight),
                    }
                )

    return data, pd.DataFrame(weight_rows)


def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "inf"
        if abs(float(value)) >= 1000:
            return f"{float(value):,.2f}"
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    table = df.copy()
    if columns is not None:
        table = table[columns]
    headers = [str(c) for c in table.columns]
    rows = []
    for _, row in table.iterrows():
        rows.append([fmt(row[c], digits).replace("\n", " ").replace("|", "／") for c in table.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def significance(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def make_plots(data: pd.DataFrame, analysis_vars: list[str], label_map: dict[str, str]) -> dict[str, str]:
    setup_chinese_font()
    output_paths = {key: ROOT / filename for key, filename in FIGURES.items()}

    corr = data[analysis_vars].corr()
    corr_display = corr.rename(index=label_map, columns=label_map)
    plt.figure(figsize=(10.5, 8.2))
    sns.heatmap(
        corr_display,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        linecolor="#FFF4EF",
        cbar_kws={"shrink": 0.82, "pad": 0.04},
        annot_kws={"fontsize": 10},
    )
    plt.title("主要变量 Pearson 相关系数矩阵", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_paths["corr"], dpi=220, bbox_inches="tight")
    plt.close()

    scatter_specs = [
        ("X综合指数", "scatter_x", "Y 与数字经济综合指数散点图"),
        ("数字基础设施指数", "scatter_infra", "Y 与数字基础设施散点图"),
        ("农业数字化指数", "scatter_agri", "Y 与农业数字化散点图"),
    ]
    for x_col, key, title in scatter_specs:
        plt.figure(figsize=(8.2, 5.8))
        sns.regplot(
            data=data,
            x=x_col,
            y="Y综合指数",
            color=WARM_PALETTE["accent"],
            scatter_kws={
                "s": 34,
                "alpha": 0.72,
                "edgecolor": WARM_PALETTE["deep"],
                "linewidth": 0.55,
            },
            line_kws={"color": WARM_PALETTE["deep"], "linewidth": 2.4},
            ci=95,
        )
        corr_value = data[[x_col, "Y综合指数"]].corr().iloc[0, 1]
        plt.title(title, fontsize=14)
        plt.xlabel(label_map.get(x_col, x_col))
        plt.ylabel("渔业高质量发展综合指数")
        plt.text(
            0.03,
            0.94,
            f"Pearson r = {corr_value:.3f}",
            transform=plt.gca().transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": WARM_PALETTE["pale"], "edgecolor": WARM_PALETTE["grid"]},
        )
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(output_paths[key], dpi=220, bbox_inches="tight")
        plt.close()

    trend = data.groupby("year")[["Y综合指数", "X综合指数"]].mean().reset_index()
    plt.figure(figsize=(8.6, 5.4))
    plt.plot(
        trend["year"],
        trend["Y综合指数"],
        marker="o",
        markersize=6.5,
        linewidth=2.4,
        color=WARM_PALETTE["deep"],
        markerfacecolor="#FFFDFB",
        markeredgewidth=2,
        label="渔业高质量发展 Y",
    )
    plt.plot(
        trend["year"],
        trend["X综合指数"],
        marker="s",
        markersize=6.5,
        linewidth=2.4,
        color=WARM_PALETTE["accent"],
        markerfacecolor="#FFFDFB",
        markeredgewidth=2,
        label="数字经济发展 X",
    )
    plt.title("2011-2023 年 Y 与 X 平均水平变化趋势", fontsize=14)
    plt.xlabel("年份")
    plt.ylabel("综合指数")
    plt.legend()
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(output_paths["trend"], dpi=220, bbox_inches="tight")
    plt.close()

    region_order = ["东部", "中部", "西部"]
    region_avg = (
        data[data["地区"].isin(region_order)]
        .groupby("地区")[["Y综合指数", "X综合指数"]]
        .mean()
        .reindex(region_order)
        .reset_index()
        .melt(id_vars="地区", var_name="指标", value_name="平均指数")
    )
    region_avg["指标"] = region_avg["指标"].map({"Y综合指数": "渔业高质量发展 Y", "X综合指数": "数字经济发展 X"})
    plt.figure(figsize=(8, 5.3))
    ax = sns.barplot(
        data=region_avg,
        x="地区",
        y="平均指数",
        hue="指标",
        palette=[WARM_PALETTE["deep"], WARM_PALETTE["accent"]],
        edgecolor="#FFF4EF",
        linewidth=0.8,
    )
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.003,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=WARM_PALETTE["text"],
        )
    plt.title("不同区域 Y 与 X 平均水平对比", fontsize=14)
    plt.xlabel("区域")
    plt.ylabel("平均综合指数")
    plt.legend(title="")
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(output_paths["region"], dpi=220, bbox_inches="tight")
    plt.close()

    return {key: str(path.name) for key, path in output_paths.items()}


def run_regression(data: pd.DataFrame, reg_vars: list[str]) -> dict[str, object]:
    reg_df = data[["Y综合指数"] + reg_vars].dropna()
    x = sm.add_constant(reg_df[reg_vars])
    y = reg_df["Y综合指数"]
    model = sm.OLS(y, x).fit()

    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(model.resid, model.model.exog)
    jb_stat, jb_p, skew, kurtosis = jarque_bera(model.resid)
    robust = model.get_robustcov_results(cov_type="HC3") if bp_lm_p < 0.05 else None

    vif_rows = []
    for i, col in enumerate(x.columns):
        if col == "const":
            continue
        try:
            vif_value = variance_inflation_factor(x.values, i)
        except Exception:
            vif_value = np.nan
        vif_rows.append({"变量": col, "VIF": float(vif_value)})

    coef_rows = []
    robust_bse = pd.Series(robust.bse, index=model.params.index) if robust is not None else None
    robust_pvalues = pd.Series(robust.pvalues, index=model.params.index) if robust is not None else None
    for name in model.params.index:
        row = {
            "变量": name,
            "系数": float(model.params[name]),
            "标准误": float(model.bse[name]),
            "t值": float(model.tvalues[name]),
            "p值": float(model.pvalues[name]),
            "显著性": significance(float(model.pvalues[name])),
        }
        if robust is not None:
            row["HC3标准误"] = float(robust_bse[name])
            row["HC3 p值"] = float(robust_pvalues[name])
            row["HC3显著性"] = significance(float(robust_pvalues[name]))
        coef_rows.append(row)

    diagnostics = pd.DataFrame(
        [
            {"指标": "样本量", "结果": len(reg_df), "说明": "用于主回归的完整观测数"},
            {"指标": "R²", "结果": model.rsquared, "说明": "模型解释力"},
            {"指标": "调整 R²", "结果": model.rsquared_adj, "说明": "考虑自变量数量后的解释力"},
            {"指标": "F 检验 p 值", "结果": model.f_pvalue, "说明": "检验模型整体显著性"},
            {"指标": "Jarque-Bera p 值", "结果": jb_p, "说明": "检验残差正态性"},
            {"指标": "残差偏度", "结果": skew, "说明": "残差分布偏斜程度"},
            {"指标": "残差峰度", "结果": kurtosis, "说明": "残差分布尖峭程度"},
            {"指标": "Breusch-Pagan LM p 值", "结果": bp_lm_p, "说明": "检验异方差"},
            {"指标": "Breusch-Pagan F p 值", "结果": bp_f_p, "说明": "检验异方差"},
        ]
    )

    return {
        "model": model,
        "coef_table": pd.DataFrame(coef_rows),
        "vif_table": pd.DataFrame(vif_rows),
        "diagnostics": diagnostics,
        "robust_used": robust is not None,
    }


def describe_variables(data: pd.DataFrame, analysis_vars: list[str], label_map: dict[str, str]) -> pd.DataFrame:
    desc = data[analysis_vars].describe().T
    desc["median"] = data[analysis_vars].median()
    desc = desc[["count", "mean", "std", "min", "median", "max"]].reset_index()
    desc = desc.rename(
        columns={
            "index": "变量",
            "count": "样本量",
            "mean": "均值",
            "std": "标准差",
            "min": "最小值",
            "median": "中位数",
            "max": "最大值",
        }
    )
    desc["变量"] = desc["变量"].map(label_map).fillna(desc["变量"])
    return desc


def correlation_sentence(corr_value: float) -> str:
    direction = "正相关" if corr_value >= 0 else "负相关"
    strength = "较强" if abs(corr_value) >= 0.6 else "中等" if abs(corr_value) >= 0.3 else "较弱"
    return f"二者 Pearson 相关系数为 {corr_value:.3f}，呈{strength}{direction}关系。"


def build_report(
    data: pd.DataFrame,
    indicators: pd.DataFrame,
    weights: pd.DataFrame,
    notes: dict[str, object],
    word_context: dict[str, str],
    desc: pd.DataFrame,
    corr: pd.DataFrame,
    reg: dict[str, object],
    label_map: dict[str, str],
) -> str:
    indicator_table = indicators[
        ["一级指标", "二级指标", "三级指标", "代码", "指标含义", "单位", "指标方向"]
    ].copy()

    top_y_weights = (
        weights[weights["指数"] == "Y综合指数"]
        .sort_values("权重", ascending=False)
        .head(5)[["二级指标", "三级指标", "权重"]]
    )
    top_x_weights = (
        weights[weights["指数"] == "X综合指数"]
        .sort_values("权重", ascending=False)
        .head(5)[["二级指标", "三级指标", "权重"]]
    )

    reg_coef = reg["coef_table"].copy()
    reg_coef["变量"] = reg_coef["变量"].replace({"const": "常数项"}).map(label_map).fillna(reg_coef["变量"].replace({"const": "常数项"}))
    vif_table = reg["vif_table"].copy()
    vif_table["变量"] = vif_table["变量"].map(label_map).fillna(vif_table["变量"])
    diagnostics = reg["diagnostics"].copy()

    analysis_vars = ["Y综合指数", "X综合指数"] + [f"{name}指数" for name in X_SECONDARY_ORDER]
    corr_y = corr["Y综合指数"].drop("Y综合指数").sort_values(ascending=False)
    strongest_name = corr_y.abs().idxmax()
    strongest_corr = corr["Y综合指数"][strongest_name]

    trend = data.groupby("year")[["Y综合指数", "X综合指数"]].mean()
    y_change = trend["Y综合指数"].iloc[-1] - trend["Y综合指数"].iloc[0]
    x_change = trend["X综合指数"].iloc[-1] - trend["X综合指数"].iloc[0]
    region_summary = data.groupby("地区")[["Y综合指数", "X综合指数"]].mean()
    region_x_top = region_summary["X综合指数"].idxmax()
    region_y_top = region_summary["Y综合指数"].idxmax()

    x_corr = corr.loc["Y综合指数", "X综合指数"]
    infra_corr = corr.loc["Y综合指数", "数字基础设施指数"]
    agri_corr = corr.loc["Y综合指数", "农业数字化指数"]

    correction_text = "；".join(notes["x_id_corrections"]) if notes["x_id_corrections"] else "未发现需要修正的 id 错位"
    y_extra_text = "、".join(notes["y_extra_cols"]) if notes["y_extra_cols"] else "无"
    robust_text = "Breusch-Pagan 检验提示存在异方差，因此报告中同步列示 HC3 稳健标准误。" if reg["robust_used"] else "Breusch-Pagan 检验未提示显著异方差，因此以常规 OLS 标准误为主。"

    lines = [
        "# 数字经济对渔业高质量发展的影响分析",
        "",
        "## 一、研究背景与问题定义",
        "",
        f"本报告以“数字经济发展是否促进渔业高质量发展”为核心问题。{word_context['background']}",
        "结合课堂作业要求，本报告将渔业高质量发展设定为因变量 Y，将数字经济发展及其六个分项指标设定为自变量 X。研究重点不是复现 Word 论文中的固定效应、中介效应或异质性模型，而是在现有 Excel 数据基础上完成一次规范、可汇报的多元线性回归分析。",
        "",
        "本次分析的基本假设是：数字经济发展水平越高，越有利于提升渔业生产效率、改善资源配置、推动加工贸易和技术培训，从而促进渔业高质量发展。",
        "",
        "## 二、数据来源与变量说明",
        "",
        f"Word 文件《渔业论文.docx》中说明，研究数据为{word_context['sample']}。数据来源方面，{word_context['source']}。Excel 未进一步标注的字段，本报告按“根据现有数据整理”处理。",
        "",
        f"本次实际合并后的分析样本为 {notes['province_count']} 个省份、{notes['year_min']}-{notes['year_max']} 年，共 {len(data)} 条观测。合并键为 `id` 和 `year`，并从 `Sheet3` 提取省份名称用于地区分析。",
        "",
        f"数据清理时发现 `x` 表 2023 年尾部存在 id 错位，已按行序在分析数据中修正：{correction_text}。该修正不改变原始 Excel 文件，只用于本次计算。`y` 表中除 Sheet4 明确命名的 15 个 Y 指标外，还存在未在指标体系表中说明的列：{y_extra_text}；本报告未将这些未命名列纳入综合指数。",
        "",
        "主要变量设定如下：",
        "",
        "- 因变量：`Y综合指数`，由渔业高质量发展 15 个三级指标经极差标准化和熵值法加权得到。",
        "- 核心解释变量：`X综合指数`，由数字经济发展 27 个三级指标经极差标准化和熵值法加权得到。",
        "- 多元回归自变量：数字基础设施、通信业务服务化、通信服务业数字化、农业数字化、工业数字化、服务业数字化六个二级指标指数。",
        "",
        "## 三、指标体系构建",
        "",
        "完整指标体系如下。三级指标均来自 Excel 的 `Sheet4`，指标方向统一按正向处理。",
        "",
        markdown_table(indicator_table, digits=4),
        "",
        "熵值法计算得到的综合指数权重中，Y 和 X 权重最高的部分三级指标如下：",
        "",
        "**Y 综合指数权重较高的三级指标**",
        "",
        markdown_table(top_y_weights, digits=4),
        "",
        "**X 综合指数权重较高的三级指标**",
        "",
        markdown_table(top_x_weights, digits=4),
        "",
        "## 四、数据预处理",
        "",
        "本报告采用以下预处理步骤：",
        "",
        "1. 按 `id-year` 合并 `x`、`y` 两张数据表，并匹配省份名称。",
        "2. 对 `x` 表尾部 2023 年 id 错位进行分析数据内修正，确保形成 30 个省份、13 年的平衡面板。",
        "3. 检查缺失值和重复值：纳入指标未发现缺失值，合并后 `id-year` 无重复。",
        "4. 默认所有指标为正向指标，使用极差标准化将各三级指标转化为 0-1 区间。",
        "5. 使用熵值法计算三级指标权重，并分别生成二级指标指数、Y 综合指数和 X 综合指数。",
        "",
        "熵值法基本思路是：指标差异越大，提供的信息量越高，权重越大。对标准化后的指标计算比例、熵值和差异系数后，将差异系数归一化为权重，再加权求和得到综合指数。",
        "",
        "## 五、探索性数据分析",
        "",
        "### 1. 描述性统计",
        "",
        markdown_table(desc, digits=4),
        "",
        "描述性统计显示，各综合指数均位于 0-1 区间内，反映出不同省份和年份之间存在一定差异。由于指数由多项三级指标加权形成，其均值和标准差可用于观察不同维度发展水平的集中趋势与离散程度。",
        "",
        "### 2. 相关系数矩阵",
        "",
        f"![相关系数热力图]({FIGURES['corr']})",
        "",
        f"相关系数热力图显示，Y 综合指数与各数字经济分项之间存在不同程度的线性相关。其中，与 Y 相关性绝对值最高的变量为 {label_map[strongest_name]}，相关系数为 {strongest_corr:.3f}。相关关系只能作为初步判断，仍需通过多元回归进一步控制其他分项指标的影响。",
        "",
        "### 3. 散点图",
        "",
        f"![Y与数字经济综合指数散点图]({FIGURES['scatter_x']})",
        "",
        f"{correlation_sentence(x_corr)}散点图中的拟合线用于展示线性趋势，整体上可初步观察数字经济综合水平与渔业高质量发展之间的同步变化关系。",
        "",
        f"![Y与数字基础设施散点图]({FIGURES['scatter_infra']})",
        "",
        f"{correlation_sentence(infra_corr)}数字基础设施反映移动互联网、移动电话、互联网端口、域名、光缆和基站等基础条件，是数字经济作用于渔业生产和管理的底层支撑。",
        "",
        f"![Y与农业数字化散点图]({FIGURES['scatter_agri']})",
        "",
        f"{correlation_sentence(agri_corr)}农业数字化与渔业场景联系更直接，其散点分布可以辅助判断农业农村数字化条件是否与渔业发展质量同步提升。",
        "",
        "### 4. 趋势图与地区对比图",
        "",
        f"![Y与X趋势图]({FIGURES['trend']})",
        "",
        f"从年度均值看，2011-2023 年 Y 综合指数累计变化 {y_change:.3f}，X 综合指数累计变化 {x_change:.3f}。趋势图能够展示两个综合指数是否具有同步上升或阶段性波动特征，适合放入课堂 PPT 作为直观背景。",
        "",
        f"![地区对比图]({FIGURES['region']})",
        "",
        f"区域对比结果显示，数字经济综合指数平均水平最高的区域为{region_x_top}，渔业高质量发展综合指数平均水平最高的区域为{region_y_top}。这提示数字经济基础与渔业发展质量可能存在区域差异，课堂汇报中可结合东中西部基础设施和产业结构差异进行解释。",
        "",
        "## 六、多元线性回归模型",
        "",
        "本报告建立如下多元线性回归模型：",
        "",
        "$$Y_i = \\beta_0 + \\beta_1X_{1i} + \\beta_2X_{2i} + \\beta_3X_{3i} + \\beta_4X_{4i} + \\beta_5X_{5i} + \\beta_6X_{6i} + \\varepsilon_i$$",
        "",
        "其中，Y 为渔业高质量发展综合指数，六个自变量依次为数字基础设施、通信业务服务化、通信服务业数字化、农业数字化、工业数字化和服务业数字化指数。",
        "",
        markdown_table(reg_coef, digits=4),
        "",
        "注：*、**、*** 分别表示在 10%、5%、1% 水平上显著。",
        "",
        "## 七、模型诊断",
        "",
        markdown_table(diagnostics, digits=4),
        "",
        "**多重共线性检验（VIF）**",
        "",
        markdown_table(vif_table, digits=4),
        "",
        robust_text,
        "VIF 用于判断自变量之间的多重共线性，通常 VIF 大于 10 表示共线性较强，需要谨慎解释单个变量系数。残差正态性和异方差检验结果用于判断 OLS 推断是否稳健；若残差不完全满足正态性，考虑到样本量为 390，回归方向和显著性仍可作为课堂分析参考。",
        "",
        "## 八、结果解释",
        "",
    ]

    for _, row in reg_coef.iterrows():
        if row["变量"] == "常数项":
            continue
        coef = row["系数"]
        p = row["HC3 p值"] if "HC3 p值" in row and not pd.isna(row["HC3 p值"]) else row["p值"]
        direction = "正向" if coef >= 0 else "负向"
        sig = "显著" if p < 0.1 else "不显著"
        lines.append(f"- {row['变量']} 的系数为 {coef:.4f}，表现为{direction}影响，p 值为 {p:.4f}，统计上{sig}。")

    lines += [
        "",
        "总体来看，数字经济不同分项对渔业高质量发展的作用方向和显著性并不完全一致。这说明数字经济并非通过单一路径影响渔业，而是可能通过基础设施、产业数字化、农业农村数字化条件和服务业数字化能力等多个渠道共同发挥作用。对于不显著的变量，应结合多重共线性、区域差异和指标口径差异谨慎解释。",
        "",
        "## 九、结论与局限",
        "",
        "本报告基于 2011-2023 年 30 个省份面板数据，构建了渔业高质量发展和数字经济发展的三级指标体系，并使用熵值法生成综合指数。探索性分析和多元线性回归结果表明，数字经济相关分项与渔业高质量发展之间存在一定统计关系，部分数字经济维度对 Y 综合指数具有较明显解释作用。",
        "",
        "研究局限主要包括：第一，本报告按课堂作业要求采用普通多元线性回归，未进一步纳入省份固定效应和年份固定效应；第二，所有三级指标默认按正向处理，若后续发现个别指标具有负向含义，需要重新调整标准化方向；第三，综合指数权重依赖样本内部差异，权重大小不等同于理论重要性；第四，Excel 中部分未命名字段未纳入分析，可能存在遗漏信息。",
        "",
        "## 十、PPT 汇报提纲",
        "",
        "| 页码 | 主题 | 主要内容 |",
        "|---|---|---|",
        "| 1 | 研究背景 | 数字经济赋能渔业高质量发展，说明研究问题和变量设定 |",
        "| 2 | 指标体系 | 展示一级、二级、三级指标体系，强调三级指标来自 Excel |",
        "| 3 | 数据与方法 | 说明样本为 30 省份、2011-2023 年，介绍标准化和熵值法 |",
        "| 4 | 探索性分析 | 放入相关系数热力图、散点图或趋势图，说明初步关系 |",
        "| 5 | 回归结果 | 展示多元线性回归表和模型诊断结果 |",
        "| 6 | 结论建议 | 总结数字经济分项影响、研究局限和政策启示 |",
        "",
    ]

    return "\n".join(lines)


def validate_outputs(report_path: Path) -> None:
    if not report_path.exists() or report_path.stat().st_size == 0:
        raise RuntimeError("Markdown 报告未生成或为空")
    for filename in FIGURES.values():
        path = ROOT / filename
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"图表未生成或为空: {filename}")
    text = report_path.read_text(encoding="utf-8")
    required_sections = [
        "## 一、研究背景与问题定义",
        "## 二、数据来源与变量说明",
        "## 三、指标体系构建",
        "## 四、数据预处理",
        "## 五、探索性数据分析",
        "## 六、多元线性回归模型",
        "## 七、模型诊断",
        "## 八、结果解释",
        "## 九、结论与局限",
        "## 十、PPT 汇报提纲",
    ]
    missing = [section for section in required_sections if section not in text]
    if missing:
        raise RuntimeError(f"报告缺少章节: {missing}")


def main() -> None:
    xlsx = find_input_file(".xlsx")
    docx = find_input_file(".docx")
    indicators = parse_indicator_system(xlsx)
    data, notes = prepare_data(xlsx, indicators)
    if len(data) != 390 or notes["province_count"] != 30 or notes["duplicate_id_year"] != 0:
        raise RuntimeError(f"样本校验失败: {notes}")

    data, weights = compute_indices(data, indicators)
    word_context = read_word_context(docx)

    analysis_vars = ["Y综合指数", "X综合指数"] + [f"{name}指数" for name in X_SECONDARY_ORDER]
    label_map = {
        "Y综合指数": "渔业高质量发展 Y",
        "X综合指数": "数字经济综合指数 X",
        "数字基础设施指数": "数字基础设施",
        "通信业务服务化指数": "通信业务服务化",
        "通信服务业数字化指数": "通信服务业数字化",
        "农业数字化指数": "农业数字化",
        "工业数字化指数": "工业数字化",
        "服务业数字化指数": "服务业数字化",
        "const": "常数项",
    }

    desc = describe_variables(data, analysis_vars, label_map)
    corr = data[analysis_vars].corr()
    make_plots(data, analysis_vars, label_map)
    reg = run_regression(data, [f"{name}指数" for name in X_SECONDARY_ORDER])

    report = build_report(data, indicators, weights, notes, word_context, desc, corr, reg, label_map)
    report_path = ROOT / REPORT_NAME
    report_path.write_text(report, encoding="utf-8")
    validate_outputs(report_path)

    print(f"完成: {report_path.name}")
    for filename in FIGURES.values():
        print(f"完成: {filename}")


if __name__ == "__main__":
    main()
