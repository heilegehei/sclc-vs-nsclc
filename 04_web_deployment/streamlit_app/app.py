from __future__ import annotations


import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_runtime import (
    FROZEN_DECISION_THRESHOLD,
    GROUP_LABELS,
    INPUT_FIELDS,
    input_defaults,
    input_template,
    model_metadata,
    score_frame,
    score_one,
    validate_required_values,
)


st.set_page_config(
    page_title="LabSignal | SCLC vs NSCLC",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _style() -> None:
    st.markdown(
        """
        <style>
          :root {--apple-black:#000;--ink:#1d1d1f;--sub:#6e6e73;--blue:#0071e3;--blue-hover:#0077ed;--grey:#f5f5f7;--line:#d2d2d7;--darksub:#a1a1a6;--green:#30d158;--orange:#ff9f0a;--display:"SF Pro Display","SF Pro Icons","Helvetica Neue",Helvetica,Arial,sans-serif;--text:"SF Pro Text","SF Pro Icons","Helvetica Neue",Helvetica,Arial,sans-serif;--ui:var(--text)}
          html {scroll-behavior:smooth;background:#fff}
          html,body,[class*="css"] {font-family:var(--text);color:var(--ink);font-size:17px;font-weight:400;letter-spacing:-.022em;line-height:1.4705882353;font-feature-settings:"kern" 1,"liga" 1;font-synthesis:none;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
          [data-testid="stMarkdownContainer"],[data-testid="stCaptionContainer"],[data-baseweb="tab"],[data-testid="stButton"] button,[data-testid="stDownloadButton"] button,[data-testid="stExpander"] summary,[data-testid="stFileUploader"] label,[data-testid="stAlert"]{font-family:var(--text)!important}
          .stApp {background:#fff}.block-container {max-width:none;padding:0 0 76px!important}.stMainBlockContainer{padding-top:0!important}
          [data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important}[data-testid="stSidebar"]{display:none}
          .globalnav{position:sticky;top:0;z-index:100;height:44px;display:flex;align-items:center;justify-content:center;background:rgba(245,245,247,.84);backdrop-filter:blur(20px) saturate(180%);border-bottom:1px solid rgba(0,0,0,.06)}
          .globalnav-inner{width:min(980px,calc(100% - 32px));display:flex;align-items:center;justify-content:space-between;gap:16px}.nav-brand{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;color:#fff;background:#1d1d1f;font-family:var(--text);font-size:10px;font-weight:600;letter-spacing:-.01em}.nav-links{display:flex;align-items:center;gap:31px}.nav-links a{color:#1d1d1f;text-decoration:none;font-family:var(--text);font-size:12px;font-weight:400;letter-spacing:-.01em;line-height:1.3333733333;transition:opacity .2s}.nav-links a:hover{opacity:.55}.nav-model{color:#6e6e73;font-family:var(--text);font-size:11px;letter-spacing:-.01em;white-space:nowrap}
          .promo{min-height:52px;display:flex;align-items:center;justify-content:center;padding:10px 18px;background:#fff;color:#1d1d1f;text-align:center;font-family:var(--text);font-size:14px;font-weight:400;letter-spacing:-.016em;line-height:1.2857742857}.promo span{color:var(--blue);margin-left:5px}
          .hero{position:relative;min-height:690px;overflow:hidden;background:#000;color:#f5f5f7;text-align:center}.hero-copy{position:relative;z-index:3;padding:58px 18px 0;animation:fade-up .9s cubic-bezier(.16,1,.3,1) both}.product-kicker{margin-bottom:8px;font-family:var(--display)!important;font-size:21px;font-weight:600;letter-spacing:.011em;line-height:1.1904761905;color:#f5f5f7}.hero h1{max-width:700px;margin:0 auto;color:#f5f5f7;font-family:var(--display)!important;font-size:clamp(42px,5vw,56px);font-weight:600;letter-spacing:-.005em;line-height:1.0714285714}.hero-sub{display:block!important;max-width:620px;margin:10px auto 0!important;color:#a1a1a6!important;text-align:center!important;font-family:var(--display)!important;font-size:clamp(19px,2vw,28px);font-weight:400;letter-spacing:.007em;line-height:1.1428571429}.hero-actions{display:flex;justify-content:center;gap:23px;margin-top:20px}.hero-actions a{color:#2997ff;text-decoration:none;font-family:var(--display)!important;font-size:21px;font-weight:400;letter-spacing:.011em;line-height:1.1904761905}.hero-actions a:after{content:' ›';font-size:21px;vertical-align:-1px}.hero-actions a:hover{text-decoration:underline}
          .hero-stage{position:absolute;inset:auto 0 0;height:376px;overflow:hidden;animation:fade-stage 1.35s .08s cubic-bezier(.16,1,.3,1) both}.hero-aura{position:absolute;left:50%;bottom:-200px;width:820px;height:510px;transform:translateX(-50%);border-radius:50%;background:radial-gradient(ellipse at center,rgba(64,156,255,.84) 0,rgba(20,94,183,.5) 24%,rgba(5,25,52,.2) 50%,rgba(0,0,0,0) 70%);filter:blur(1px);animation:aura-shift 10s ease-in-out infinite}.signal-plane{position:absolute;left:50%;bottom:-40px;width:min(810px,86vw);height:230px;transform:translateX(-50%) perspective(800px) rotateX(57deg);border:1px solid rgba(202,232,255,.32);border-radius:36px;background:linear-gradient(137deg,rgba(162,215,255,.4),rgba(30,124,223,.13) 45%,rgba(0,0,0,.07));box-shadow:0 -14px 68px rgba(52,164,255,.27),inset 0 1px 0 rgba(255,255,255,.42);animation:plane-float 8s ease-in-out infinite}.signal-plane:before{content:'';position:absolute;inset:17px;border:1px solid rgba(225,244,255,.2);border-radius:25px}.signal-core{position:absolute;left:50%;bottom:109px;width:236px;height:236px;transform:translateX(-50%);border-radius:50%;background:radial-gradient(circle at 34% 25%,#fff 0,rgba(224,243,255,.98) 10%,rgba(83,173,255,.9) 28%,rgba(11,84,184,.72) 54%,rgba(2,10,29,.2) 72%,transparent 73%);box-shadow:0 0 54px rgba(46,152,255,.75),inset -25px -31px 37px rgba(0,19,63,.33);animation:core-float 7s ease-in-out infinite}.signal-core:after{content:'';position:absolute;left:33%;top:19%;width:23%;height:16%;border-radius:50%;background:rgba(255,255,255,.86);filter:blur(8px)}.stage-spec{position:absolute;left:50%;bottom:24px;transform:translateX(-50%);color:rgba(255,255,255,.72);font-size:12px;letter-spacing:.035em;white-space:nowrap}.stage-spec b{color:#fff;font-weight:600}
          .content-tile{padding:100px 18px}.content-tile.alt{background:var(--grey)}.tile-intro{max-width:760px;margin:0 auto 45px;text-align:center}.tile-eyebrow{margin:0 0 8px;color:#6e6e73;font-family:var(--display)!important;font-size:21px;font-weight:600;letter-spacing:.011em;line-height:1.1904761905}.tile-intro h2{margin:0;color:#1d1d1f;font-family:var(--display)!important;font-size:clamp(34px,4.3vw,48px);font-weight:600;letter-spacing:-.002em;line-height:1.0834933333}.tile-intro p{max-width:650px;margin:13px auto 0;color:#6e6e73;font-family:var(--display)!important;font-size:21px;font-weight:400;letter-spacing:.011em;line-height:1.1904761905}
          .spec-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));width:min(940px,100%);margin:0 auto;border-top:1px solid #d2d2d7;border-bottom:1px solid #d2d2d7}.spec{padding:23px 27px 25px;text-align:center}.spec+.spec{border-left:1px solid #d2d2d7}.spec-label{display:block;color:#6e6e73;font-size:14px;letter-spacing:-.01em}.spec-value{display:block;margin-top:7px;color:#1d1d1f;font-size:32px;font-weight:600;letter-spacing:-.045em}.spec-caption{display:block;margin-top:4px;color:#86868b;font-size:12px;line-height:1.25}
          .workflow-anchor{display:block;position:relative;top:-44px;visibility:hidden}.input-intro{padding:100px 18px 45px;background:var(--grey);text-align:center}.input-intro .tile-intro{margin-bottom:0}.form-note{margin:0 0 23px;color:#6e6e73;font-family:var(--text);font-size:14px;font-weight:400;letter-spacing:-.016em;line-height:1.2857742857;text-align:center}.form-note b{color:#1d1d1f;font-weight:600}
          div[data-testid="stVerticalBlockBorderWrapper"]{width:min(1010px,calc(100% - 22px));margin:0 auto!important;border:0!important;border-radius:18px!important;background:#fff!important;box-shadow:0 2px 7px rgba(0,0,0,.04)!important}div[data-testid="stVerticalBlockBorderWrapper"]>div{border:0!important;background:transparent!important}
          [data-testid="stForm"]{width:100%;margin:0!important;padding:0 32px 34px!important;border:0!important;border-radius:0!important;background:transparent!important;box-shadow:none!important}
          .stTabs [data-baseweb="tab-list"]{justify-content:center;gap:20px;border-bottom:1px solid #d2d2d7}.stTabs [data-baseweb="tab"]{height:43px;padding:0 2px;border-radius:0;color:#6e6e73;background:transparent;font-family:var(--text);font-size:15px;font-weight:400;letter-spacing:-.016em;line-height:1.2857742857}.stTabs [aria-selected="true"]{color:#1d1d1f!important;font-weight:600!important}.stTabs [data-baseweb="tab-highlight"]{height:2px!important;background:#1d1d1f!important}
          [data-testid="stNumberInput"]{margin-bottom:18px}[data-testid="stNumberInput"] label{color:#1d1d1f!important;font-family:var(--text)!important;font-size:14px!important;font-weight:400!important;letter-spacing:-.016em!important;line-height:1.2857742857!important}[data-testid="stNumberInput"] input{padding:11px 12px!important;border:1px solid #d2d2d7!important;border-radius:10px!important;background:#fff!important;color:#1d1d1f!important;font-family:var(--text)!important;font-size:15px!important;font-weight:400!important;letter-spacing:-.016em!important;font-variant-numeric:tabular-nums!important}[data-testid="stNumberInput"] input:focus{border-color:#0071e3!important;box-shadow:0 0 0 3px rgba(0,113,227,.16)!important}
          .stButton>button,.stDownloadButton>button{min-height:42px;padding:0 17px!important;border:1px solid var(--blue)!important;border-radius:980px!important;background:var(--blue)!important;color:#fff!important;box-shadow:none!important;font-family:var(--text)!important;font-size:14px!important;font-weight:400!important;letter-spacing:-.016em!important;line-height:1.2857742857!important;transition:background .18s,transform .18s!important}.stButton>button:hover,.stDownloadButton>button:hover{background:var(--blue-hover)!important;transform:scale(.98)}.stButton>button:active,.stDownloadButton>button:active{transform:scale(.96)}
          .safety{width:min(1010px,calc(100% - 22px));margin:14px auto 0;color:#6e6e73;font-family:var(--text);text-align:center;font-size:12px;font-weight:400;letter-spacing:-.01em;line-height:1.3333733333}.safety b{color:#1d1d1f;font-weight:600}
          .result-tile{padding:94px 18px 86px;background:#000;color:#f5f5f7;text-align:center;animation:result-in .8s cubic-bezier(.16,1,.3,1) both}.result-tile .tile-eyebrow{color:#f5f5f7}.result-tile .tile-intro h2{color:#f5f5f7}.result-tile .tile-intro p{color:#a1a1a6}.result-grid{display:grid;grid-template-columns:1.12fr .88fr;gap:0;width:min(900px,100%);margin:0 auto;text-align:left;border-top:1px solid rgba(255,255,255,.32);border-bottom:1px solid rgba(255,255,255,.32)}.prob-display{min-height:244px;padding:30px 34px;border-right:1px solid rgba(255,255,255,.32)}.prob-label{color:#a1a1a6;font-family:var(--text);font-size:13px;font-weight:400;letter-spacing:-.01em;line-height:1.3333733333}.prob-number{margin-top:8px;color:#fff;font-family:var(--display);font-size:clamp(66px,9vw,112px);font-weight:600;letter-spacing:-.005em;line-height:1.0714285714}.prob-number span{font-size:.34em;vertical-align:top;letter-spacing:-.005em}.prob-rail{position:relative;height:4px;margin:26px 0 8px;border-radius:4px;background:rgba(255,255,255,.3)}.prob-fill{height:4px;border-radius:4px;background:#f5f5f7}.threshold{position:absolute;top:-5px;width:1px;height:14px;background:#30d158}.threshold-label{position:absolute;top:14px;transform:translateX(-38%);color:#a1a1a6;font-family:var(--text);font-size:11px;letter-spacing:-.01em;line-height:1.3333733333;white-space:nowrap}.prob-copy{margin:20px 0 0;color:#a1a1a6;font-family:var(--text);font-size:13px;font-weight:400;letter-spacing:-.01em;line-height:1.3333733333}.outcome-display{display:flex;flex-direction:column;justify-content:space-between;min-height:244px;padding:30px 34px}.outcome-label{color:#a1a1a6;font-family:var(--text);font-size:13px;font-weight:400;letter-spacing:-.01em;line-height:1.3333733333}.outcome-main{margin-top:9px;color:#fff;font-family:var(--display);font-size:clamp(28px,4vw,43px);font-weight:600;letter-spacing:-.002em;line-height:1.0834933333}.outcome-main.low{color:#30d158}.outcome-main.high{color:#ff9f0a}.outcome-delta{padding-top:18px;border-top:1px solid rgba(255,255,255,.18);color:#a1a1a6;font-family:var(--text);font-size:12px;font-weight:400;letter-spacing:-.01em;line-height:1.3333733333}.outcome-delta b{display:block;margin-top:4px;color:#f5f5f7;font-family:var(--display);font-size:25px;font-weight:600;letter-spacing:.007em;line-height:1.1428571429}
          .index-band{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));width:min(1050px,100%);margin:48px auto 0;border-top:1px solid #d2d2d7;border-bottom:1px solid #d2d2d7;background:#fff;color:#1d1d1f}.index{min-height:126px;padding:19px 12px;text-align:center}.index+.index{border-left:1px solid #d2d2d7}.index-name{color:#6e6e73;font-family:var(--text);font-size:13px;font-weight:400;letter-spacing:-.01em}.index-value{display:block;margin-top:8px;font-family:var(--display);font-size:28px;font-weight:600;letter-spacing:.007em;line-height:1.1428571429}.index-formula{display:block;margin-top:7px;color:#86868b;font-family:var(--text);font-size:10px;letter-spacing:-.01em;line-height:1.3333733333}
          .chart-wrap{width:min(1040px,calc(100% - 36px));margin:0 auto;padding:26px 25px 10px;border-top:1px solid #d2d2d7;border-bottom:1px solid #d2d2d7}.chart-title{margin-bottom:5px;text-align:center;color:#1d1d1f;font-family:var(--display);font-size:28px;font-weight:600;letter-spacing:.007em;line-height:1.1428571429}.chart-copy{margin:0 auto 22px;color:#6e6e73;font-family:var(--text);text-align:center;font-size:14px;font-weight:400;letter-spacing:-.016em;line-height:1.2857742857}.chart-wrap [data-testid="stPlotlyChart"]{margin-top:0!important}.chart-wrap .js-plotly-plot{margin:auto!important}
          [data-testid="stExpander"]{width:min(1040px,calc(100% - 36px));margin:20px auto 0;border:1px solid #d2d2d7!important;border-radius:0!important;background:#fff!important}[data-testid="stExpander"] summary{color:#1d1d1f!important;font-family:var(--text)!important;font-size:14px!important;font-weight:400!important;letter-spacing:-.016em!important;line-height:1.2857742857!important}[data-testid="stDataFrame"]{border:1px solid #d2d2d7;border-radius:0!important;overflow:hidden}.download-row{width:min(1040px,calc(100% - 36px));margin:20px auto 0;text-align:center}
          .batch-box{width:min(1010px,calc(100% - 36px));margin:0 auto;border-top:1px solid #d2d2d7;border-bottom:1px solid #d2d2d7}.batch-box [data-testid="stExpander"]{width:100%;margin:0;border:0!important}.footer{padding:42px 18px;background:#f5f5f7;color:#6e6e73;font-family:var(--text);text-align:center;font-size:12px;font-weight:400;letter-spacing:-.01em;line-height:1.3333733333}.footer b{color:#1d1d1f;font-weight:600}
          @keyframes fade-up{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}@keyframes fade-stage{from{opacity:0;transform:translateY(58px) scale(.97)}to{opacity:1;transform:none}}@keyframes core-float{0%,100%{transform:translateX(-50%) translateY(0)}50%{transform:translateX(-50%) translateY(-8px)}}@keyframes plane-float{0%,100%{transform:translateX(-50%) perspective(800px) rotateX(57deg) translateY(0)}50%{transform:translateX(-50%) perspective(800px) rotateX(57deg) translateY(-5px)}}@keyframes aura-shift{0%,100%{filter:blur(1px);opacity:.9}50%{filter:blur(4px);opacity:1}}@keyframes result-in{from{opacity:0;transform:translateY(25px)}to{opacity:1;transform:none}}
          @media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important}}
          @media(max-width:760px){.nav-links{gap:15px}.nav-links a:nth-child(n+4){display:none}.nav-model{display:none}.promo{min-height:48px;font-size:12px}.hero{min-height:610px}.hero-copy{padding-top:48px}.hero-stage{height:330px}.content-tile,.input-intro{padding:74px 14px}.tile-intro p{font-size:18px}.spec-row{grid-template-columns:1fr}.spec+.spec{border-top:1px solid #d2d2d7;border-left:0}div[data-testid="stVerticalBlockBorderWrapper"],[data-testid="stForm"]{width:calc(100% - 20px);padding-right:18px!important;padding-left:18px!important}.stTabs [data-baseweb="tab-list"]{gap:14px}.result-grid{grid-template-columns:1fr}.prob-display{border-right:0;border-bottom:1px solid rgba(255,255,255,.32)}.index-band{grid-template-columns:repeat(2,minmax(0,1fr))}.index+.index{border-left:0}.index:nth-child(even){border-left:1px solid #d2d2d7}.index:nth-child(n+3){border-top:1px solid #d2d2d7}.chart-wrap,.batch-box,[data-testid="stExpander"],.download-row{width:calc(100% - 28px)}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _compact(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _render_nav(metadata: dict) -> None:
    st.markdown(
        f"""
        <nav class="globalnav"><div class="globalnav-inner">
          <span class="nav-brand">LS</span>
          <div class="nav-links"><a href="#overview">Overview</a><a href="#assessment">Assessment</a><a href="#model">Model</a><a href="#batch">Batch</a></div>
          <span class="nav-model">Weighted voting · {metadata['member_n']} learners</span>
        </div></nav>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(metadata: dict) -> None:
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-copy">
            <div class="product-kicker">LabSignal</div>
            <h1>SCLC versus NSCLC.</h1>
            <p class="hero-sub">A frozen laboratory model. One clear inference pathway.</p>
            <div class="hero-actions"><a href="#assessment">Start assessment</a><a href="#overview">Explore model</a></div>
          </div>
          <div class="hero-stage"><div class="hero-aura"></div><div class="signal-plane"></div><div class="signal-core"></div><div class="stage-spec"><b>{len(metadata['selected_direct']) + len(metadata['fixed_composites'])} variables</b>&nbsp;&nbsp;·&nbsp;&nbsp;{metadata['member_n']} fixed base learners&nbsp;&nbsp;·&nbsp;&nbsp;threshold {metadata['threshold']:.4f}</div></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(metadata: dict) -> None:
    st.markdown(
        f"""
        <span id="overview" class="workflow-anchor"></span><section class="content-tile">
          <div class="tile-intro"><div class="tile-eyebrow">LabSignal model</div><h2>Designed for a precise hand-off.</h2><p>Enter routine laboratory measurements, calculate the prespecified indices, and apply the same saved weighted-voting inference path every time.</p></div>
          <div class="spec-row">
            <div class="spec"><span class="spec-label">Final model variables</span><span class="spec-value">{len(metadata['selected_direct']) + len(metadata['fixed_composites'])}</span><span class="spec-caption">13 direct predictors + 5 composite indices</span></div>
            <div class="spec"><span class="spec-label">Base learners</span><span class="spec-value">{metadata['member_n']}</span><span class="spec-caption">Fixed voting weights sum to 1.00</span></div>
            <div class="spec"><span class="spec-label">Decision threshold</span><span class="spec-value">{metadata['threshold']:.4f}</span><span class="spec-caption">Frozen before interface use</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_inputs() -> tuple[dict[str, float], bool]:
    defaults = input_defaults()
    values: dict[str, float] = {}
    groups = {group: [field for field in INPUT_FIELDS if field["group"] == group] for group in GROUP_LABELS}

    st.markdown(
        """
        <span id="assessment" class="workflow-anchor"></span>
        <div class="input-intro"><div class="tile-intro"><div class="tile-eyebrow">Assessment</div><h2>Enter the measurements.</h2><p>Use the three clinical domains below. Starting values are only frozen development-set medians; replace them with the individual laboratory results.</p></div></div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            "<p class='form-note'><b>*</b> Final direct predictor &nbsp;&nbsp; <b>†</b> Component used to calculate a prespecified composite index</p>",
            unsafe_allow_html=True,
        )
        with st.form("prediction_form", border=False):
            tabs = st.tabs([f"{group} · {label}" for group, label in GROUP_LABELS.items()])
            for tab, (group, fields) in zip(tabs, groups.items()):
                with tab:
                    columns = st.columns(2, gap="large")
                    for index, item in enumerate(fields):
                        feature = item["feature"]
                        marker = "*" if item["kind"] == "selected" else "†"
                        values[feature] = columns[index % 2].number_input(
                            f"{feature}{marker} — {item['label']} ({item['unit']})",
                            min_value=0.0,
                            value=float(defaults[feature]),
                            step=max(float(defaults[feature]) / 100, 0.01),
                            format="%.4f",
                            help="* final direct predictor; † component used to calculate a pre-specified composite index.",
                            key=f"input_{feature}",
                        )
            submitted = st.form_submit_button("Generate prediction", type="primary")
    st.markdown(
        "<p class='safety'><b>Important:</b> This is a research-model output and must be interpreted with clinical, pathological, and imaging evidence. The interface does not save entered laboratory values.</p>",
        unsafe_allow_html=True,
    )
    return values, submitted


def _contribution_chart(contribution: pd.DataFrame) -> go.Figure:
    display = contribution.sort_values("Weighted contribution", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=display["Weighted contribution"],
            y=display["Model"],
            orientation="h",
            marker_color="#1d1d1f",
            hovertemplate="%{y}<br>Weighted contribution: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=450,
        margin=dict(l=4, r=8, t=4, b=42),
        xaxis_title="Contribution to weighted probability",
        yaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "SF Pro Text, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif", "color": "#1d1d1f", "size": 12},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8e8ed", zeroline=False, tickformat=".3f")
    figure.update_yaxes(showgrid=False, automargin=True)
    return figure


def _render_result(values: dict[str, float]) -> None:
    issues = validate_required_values(values)
    if issues:
        for issue in issues:
            st.error(issue)
        return

    with st.spinner("Applying the frozen preprocessing and ensemble model…"):
        result, contribution, audit = score_one(values)

    probability = float(result["weighted_voting_probability"])
    predicted_label = str(result["model_label"])
    above = probability >= FROZEN_DECISION_THRESHOLD
    output_text = "Predicted SCLC" if above else "Predicted NSCLC"
    output_class = "high" if above else "low"
    probability_pct = probability * 100
    threshold_pct = FROZEN_DECISION_THRESHOLD * 100
    delta = probability - FROZEN_DECISION_THRESHOLD
    composites = [
        ("SIRI", float(result["calculated_SIRI"]), "NEUT# × MONO# / LYMPH#"),
        ("LMR", float(result["calculated_LMR"]), "LYMPH# / MONO#"),
        ("GAR", float(result["calculated_GAR"]), "GLU / ALB"),
        ("PNI", float(result["calculated_PNI"]), "ALB + 5 × LYMPH#"),
        ("HALP", float(result["calculated_HALP"]), "HGB × ALB × LYMPH# / PLT"),
    ]

    st.markdown(
        f"""
        <section class="result-tile"><div class="tile-intro"><div class="tile-eyebrow">Prediction</div><h2>One probability. Clearly shown.</h2><p>The saved preprocessor and fixed weighted-voting ensemble generated this result.</p></div>
          <div class="result-grid">
            <div class="prob-display"><div class="prob-label">PREDICTED SCLC PROBABILITY</div><div class="prob-number">{probability_pct:.1f}<span>%</span></div><div class="prob-rail"><div class="prob-fill" style="width:{max(.25, min(probability_pct, 100)):.3f}%"></div><div class="threshold" style="left:{threshold_pct:.3f}%"></div><div class="threshold-label" style="left:{threshold_pct:.3f}%">threshold {FROZEN_DECISION_THRESHOLD:.4f}</div></div><p class="prob-copy">Probability on a 0–100% scale. The green marker denotes the frozen development-set threshold.</p></div>
            <div class="outcome-display"><div><div class="outcome-label">MODEL OUTPUT</div><div class="outcome-main {output_class}">{html.escape(output_text)}</div></div><div class="outcome-delta">Probability minus threshold<b>{delta:+.3f}</b></div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    cards = "".join(
        f"<div class='index'><span class='index-name'>{name}</span><span class='index-value'>{_compact(value)}</span><span class='index-formula'>{formula}</span></div>"
        for name, value, formula in composites
    )
    st.markdown(
        f"""
        <section class="content-tile"><span id="model" class="workflow-anchor"></span><div class="tile-intro"><div class="tile-eyebrow">Model composition</div><h2>Calculated from the same input.</h2><p>Five prespecified composite indices are generated before the frozen model produces the final probability.</p></div><div class="index-band">{cards}</div>
        """,
        unsafe_allow_html=True,
    )
    if audit["clipped_low_n"] or audit["clipped_high_n"]:
        st.info(
            f"Frozen preprocessing clipped {audit['clipped_low_n'] + audit['clipped_high_n']} value(s) "
            "to its development-set limits before model scoring."
        )
    st.markdown(
        """<div class="chart-wrap"><div class="chart-title">Weighted contribution</div><p class="chart-copy">Each fixed base learner’s contribution to the displayed probability.</p>""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_contribution_chart(contribution), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("View base-model probabilities and fixed weights", expanded=False):
        display = contribution.copy()
        display["Weight"] = display["Weight"].map(lambda value: f"{value:.2%}")
        display["Probability"] = display["Probability"].map(lambda value: f"{value:.4f}")
        display["Weighted contribution"] = display["Weighted contribution"].map(lambda value: f"{value:.4f}")
        st.dataframe(display, hide_index=True, use_container_width=True)
    summary = pd.DataFrame(
        [{
            "weighted_voting_probability": probability,
            "frozen_decision_threshold": FROZEN_DECISION_THRESHOLD,
            "model_label": predicted_label,
            **{item["feature"]: values[item["feature"]] for item in INPUT_FIELDS},
            **{f"calculated_{name}": value for name, value, _ in composites},
        }]
    )
    st.markdown("<div class='download-row'>", unsafe_allow_html=True)
    st.download_button(
        "Download prediction summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="weighted_voting_prediction.csv",
        mime="text/csv",
    )
    st.markdown("</div></section>", unsafe_allow_html=True)


def _render_batch() -> None:
    st.markdown(
        """
        <span id="batch" class="workflow-anchor"></span><section class="content-tile alt"><div class="tile-intro"><div class="tile-eyebrow">Batch</div><h2>A cohort view, when you need it.</h2><p>Use a prepared CSV to apply the same saved inference path to multiple records. <code>record_id</code> is optional and is returned unchanged.</p></div><div class="batch-box">
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Open batch assessment", expanded=False):
        st.download_button(
            "Download input template",
            data=input_template().to_csv(index=False).encode("utf-8-sig"),
            file_name="weighted_voting_input_template.csv",
            mime="text/csv",
        )
        upload = st.file_uploader("CSV input", type=["csv"], key="batch_upload")
        if upload is not None:
            try:
                source = pd.read_csv(upload)
                required = [item["feature"] for item in INPUT_FIELDS]
                missing = [feature for feature in required if feature not in source.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                    return
                scored, _, audit = score_frame(source)
                st.success(f"Scored {len(scored):,} record(s).")
                st.dataframe(scored.head(20), hide_index=True, use_container_width=True)
                st.caption(
                    "Frozen preprocessing audit: "
                    f"{audit['clipped_low_n'] + audit['clipped_high_n']} clipped value(s); "
                    "non-selected direct variables are completed by the saved preprocessor."
                )
                st.download_button(
                    "Download batch predictions (CSV)",
                    data=scored.to_csv(index=False).encode("utf-8-sig"),
                    file_name="weighted_voting_batch_predictions.csv",
                    mime="text/csv",
                )
            except Exception as error:
                st.error(f"Batch scoring could not be completed: {error}")
    st.markdown("</div></section>", unsafe_allow_html=True)


def _render_footer(metadata: dict) -> None:
    st.markdown(
        f"""<footer class="footer"><b>LabSignal</b> &nbsp;·&nbsp; Frozen weighted-voting inference interface &nbsp;·&nbsp; Artifact SHA-256: {metadata['artifact_sha256'][:16]}…<br>This interface does not write entered laboratory values to the project dataset or retrain the model.</footer>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    _style()
    metadata = model_metadata()
    _render_nav(metadata)
    _render_hero(metadata)
    _render_overview(metadata)
    values, submitted = _render_inputs()
    if submitted:
        _render_result(values)
    _render_batch()
    _render_footer(metadata)


if __name__ == "__main__":
    main()
