"""
Streamlit frontend for Cross-Domain Federated Transfer Learning
Satellite Image Classification — ResNet-50 / EuroSAT / FedAvg
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Paths & constants ───────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
CENTRALIZED_PATH  = BASE_DIR / "Colab Results" / "resnet50_eurosat.pth"
FEDERATED_PATH    = BASE_DIR / "Colab Results" / "resnet50_eurosat_federated.pth"

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]
CLASS_EMOJIS = {
    "AnnualCrop":            "🌾",
    "Forest":                "🌲",
    "HerbaceousVegetation":  "🌿",
    "Highway":               "🛣️",
    "Industrial":            "🏭",
    "Pasture":               "🐄",
    "PermanentCrop":         "🍇",
    "Residential":           "🏘️",
    "River":                 "🌊",
    "SeaLake":               "🏖️",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove default top padding */
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] section {
        border: 2px dashed rgba(99,179,237,0.4) !important;
        border-radius: 10px !important;
        background: rgba(99,179,237,0.04) !important;
    }

    /* Callout agreement */
    .agree-box {
        background: rgba(104,211,145,0.12);
        border: 1px solid rgba(104,211,145,0.4);
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 8px;
        font-size: 1.05rem;
        color: #68d391;
    }
    .disagree-box {
        background: rgba(252,129,129,0.12);
        border: 1px solid rgba(252,129,129,0.4);
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 8px;
        font-size: 1.05rem;
        color: #fc8181;
    }
</style>
""", unsafe_allow_html=True)

# ─── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models — please wait…")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(path: Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = models.resnet50(weights=None)
        for p in m.parameters():
            p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, 10)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval().to(device)
        return m, ckpt

    c_model, c_ckpt = _build(CENTRALIZED_PATH)
    f_model, f_ckpt = _build(FEDERATED_PATH)
    return c_model, f_model, c_ckpt, f_ckpt, device


# ─── Inference ───────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, image: Image.Image, device) -> np.ndarray:
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).cpu().squeeze().numpy()
    return probs


# ─── UI components ───────────────────────────────────────────────────────────
def render_prediction_card(title: str, border_color: str, accent: str, probs: np.ndarray):
    """Render a card with the top prediction and a horizontal probability bar chart."""
    top_idx   = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_conf  = probs[top_idx] * 100

    # Sort classes by probability (descending) for the chart
    order         = np.argsort(probs)[::-1]
    sorted_labels = [f"{CLASS_EMOJIS[CLASS_NAMES[i]]} {CLASS_NAMES[i]}" for i in order]
    sorted_probs  = probs[order] * 100
    bar_colors    = [accent if j == 0 else "#374151" for j in range(len(order))]

    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{p:.1f}%" for p in sorted_probs],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(range=[0, 115], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        showlegend=False,
    )

    # Top-prediction summary header
    st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {border_color}18, rgba(26,32,44,0.8));
            border: 1px solid {border_color}55;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 6px;
        '>
            <div style='color:{border_color}; font-size:0.85rem; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.08em;'>
                {title}
            </div>
            <div style='display:flex; align-items:center; gap:14px; margin-top:10px;'>
                <span style='font-size:2.8rem; line-height:1;'>{CLASS_EMOJIS[top_class]}</span>
                <div>
                    <div style='font-size:1.35rem; font-weight:700; color:#f7fafc;'>
                        {top_class}
                    </div>
                    <div style='font-size:1.05rem; color:{border_color}; font-weight:600;'>
                        {top_conf:.1f}% confidence
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_training_curves(c_ckpt: dict, f_ckpt: dict):
    """Combined validation accuracy plot for all experiments."""
    c_hist   = c_ckpt["history"]                    # 10-epoch centralized
    fc_hist  = f_ckpt["centralized_history"]         # 20-epoch centralized (from fed notebook)
    iid_hist = f_ckpt["fed_iid_history"]
    nid_hist = f_ckpt["fed_noniid_history"]

    ep10 = list(range(1, len(c_hist["val_acc"]) + 1))
    ep20 = list(range(1, len(fc_hist["val_acc"]) + 1))
    rnd  = list(range(1, len(iid_hist["global_val_acc"]) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ep10, y=c_hist["val_acc"],
        name="Centralized — 10 ep",
        mode="lines+markers",
        line=dict(color="#90cdf4", width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="Round %{x}<br>Val Acc: %{y:.2f}%<extra>Centralized 10ep</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ep20, y=fc_hist["val_acc"],
        name="Centralized — 20 ep",
        mode="lines+markers",
        line=dict(color="#4299e1", width=2.5),
        marker=dict(size=6),
        hovertemplate="Round %{x}<br>Val Acc: %{y:.2f}%<extra>Centralized 20ep</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rnd, y=iid_hist["global_val_acc"],
        name="Federated IID — 5 clients",
        mode="lines+markers",
        line=dict(color="#68d391", width=2.5),
        marker=dict(size=6),
        hovertemplate="Round %{x}<br>Val Acc: %{y:.2f}%<extra>Federated IID</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rnd, y=nid_hist["global_val_acc"],
        name="Federated Non-IID — α=0.5",
        mode="lines+markers",
        line=dict(color="#fc8181", width=2.5),
        marker=dict(size=6),
        hovertemplate="Round %{x}<br>Val Acc: %{y:.2f}%<extra>Federated Non-IID</extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Validation Accuracy Across All Experiments",
            font=dict(size=14, color="#e2e8f0"),
        ),
        xaxis=dict(
            title="Epoch / Communication Round",
            gridcolor="#2d3748",
            color="#a0aec0",
        ),
        yaxis=dict(
            title="Val Accuracy (%)",
            gridcolor="#2d3748",
            color="#a0aec0",
            range=[70, 100],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.42,
            xanchor="center", x=0.5,
            font=dict(size=11, color="#cbd5e0"),
        ),
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.5)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_model_comparison_bar(c_ckpt: dict, f_ckpt: dict):
    """Horizontal bar comparing final best val accuracies."""
    labels = [
        "Centralized (10 ep)",
        "Centralized (20 ep)",
        "Federated Non-IID (α=0.5)",
        "Federated IID (5 clients)",
    ]
    values = [
        max(c_ckpt["history"]["val_acc"]),
        max(f_ckpt["centralized_history"]["val_acc"]),
        max(f_ckpt["fed_noniid_history"]["global_val_acc"]),
        max(f_ckpt["fed_iid_history"]["global_val_acc"]),
    ]
    colors = ["#90cdf4", "#4299e1", "#fc8181", "#68d391"]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=70, t=10, b=10),
        xaxis=dict(range=[88, 100], showgrid=True, gridcolor="#2d3748",
                   color="#a0aec0", ticksuffix="%"),
        yaxis=dict(showgrid=False, color="#e2e8f0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.5)",
        font=dict(color="white", size=12),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─── Main app ────────────────────────────────────────────────────────────────
def main():
    c_model, f_model, c_ckpt, f_ckpt, device = load_models()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
        <div style='text-align:center; padding-bottom:8px;'>
            <h1 style='font-size:2.3rem; margin-bottom:4px; color:#f7fafc;'>
                🛰️ Satellite Image Classifier
            </h1>
            <p style='color:#718096; font-size:1rem; margin:0;'>
                Cross-Domain Federated Transfer Learning &nbsp;·&nbsp;
                ResNet-50 &nbsp;·&nbsp; EuroSAT RGB &nbsp;·&nbsp; 10 Land-Use Classes
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Key metrics row ───────────────────────────────────────────────────────
    c10   = max(c_ckpt["history"]["val_acc"])
    c20   = max(f_ckpt["centralized_history"]["val_acc"])
    fi    = max(f_ckpt["fed_iid_history"]["global_val_acc"])
    fn    = max(f_ckpt["fed_noniid_history"]["global_val_acc"])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "🔵 Centralized (10 ep)",
            f"{c10:.2f}%",
            help="Standard centralized training — resnet50_eurosat.pth",
        )
    with m2:
        st.metric(
            "🔷 Centralized (20 ep)",
            f"{c20:.2f}%",
            f"{c20 - c10:+.2f}% vs 10-ep baseline",
            help="Same centralized setup, trained for 20 epochs in federated notebook",
        )
    with m3:
        st.metric(
            "🟢 Federated IID",
            f"{fi:.2f}%",
            f"{fi - c20:+.2f}% vs centralized",
            help="FedAvg · 5 clients · 20 rounds · IID split — resnet50_eurosat_federated.pth",
        )
    with m4:
        st.metric(
            "🔴 Federated Non-IID",
            f"{fn:.2f}%",
            f"{fn - fi:+.2f}% vs federated IID",
            help="FedAvg · 5 clients · 20 rounds · Dirichlet α=0.5 heterogeneous split",
        )

    st.divider()

    # ── Comparison bar + Training curves ─────────────────────────────────────
    col_bar, col_curves = st.columns([1, 1.6], gap="large")

    with col_bar:
        st.subheader("Best Validation Accuracy")
        render_model_comparison_bar(c_ckpt, f_ckpt)

        # Key takeaway callout
        st.markdown("""
            <div style='
                background: rgba(104,211,145,0.08);
                border-left: 3px solid #68d391;
                border-radius: 0 8px 8px 0;
                padding: 10px 14px;
                margin-top: 4px;
                font-size: 0.9rem;
                color: #a0aec0;
            '>
                <b style='color:#68d391;'>Key finding:</b> Federated IID matches or exceeds
                centralized training while keeping raw images private.
                Non-IID data heterogeneity causes a ~1.6% accuracy drop vs IID.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
            <div style='font-size:0.85rem; color:#718096; line-height:1.7;'>
                <b style='color:#a0aec0;'>Architecture:</b> ResNet-50 frozen backbone +
                trainable FC head (20,490 params)<br>
                <b style='color:#a0aec0;'>Federated config:</b> 5 clients · 20 rounds ·
                5 local epochs · FedAvg<br>
                <b style='color:#a0aec0;'>Privacy cost:</b> 80 KB FC weights per client/round
                (15.6 MB total)<br>
                <b style='color:#a0aec0;'>Dataset:</b> EuroSAT RGB · 27,000 images ·
                80/20 train-val split
            </div>
        """, unsafe_allow_html=True)

    with col_curves:
        st.subheader("Training Curves")
        render_training_curves(c_ckpt, f_ckpt)

    st.divider()

    # ── Image upload & inference ──────────────────────────────────────────────
    st.subheader("Try It — Upload a Satellite Image")

    up_col, info_col = st.columns([1.2, 1], gap="large")

    with info_col:
        st.markdown("""
            <div style='
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px;
                padding: 16px 18px;
                font-size: 0.9rem;
                color: #a0aec0;
                line-height: 1.8;
            '>
                <b style='color:#e2e8f0; font-size:0.95rem;'>Supported classes</b><br>
        """ + "".join(
            f"<span style='margin-right:12px;'>{CLASS_EMOJIS[n]} {n}</span>"
            for n in CLASS_NAMES
        ) + """
            </div>
        """, unsafe_allow_html=True)

    with up_col:
        uploaded = st.file_uploader(
            "Drop or click to upload a satellite image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

    # ── Predictions (shown below when image is uploaded) ──────────────────────
    if uploaded is not None:
        with st.spinner("Running both models…"):
            c_probs = predict(c_model, image, device)
            f_probs = predict(f_model, image, device)

        st.divider()
        st.subheader("Model Predictions")

        pred_l, pred_r = st.columns(2, gap="large")
        with pred_l:
            render_prediction_card(
                title="Centralized ResNet-50",
                border_color="#4299e1",
                accent="#4299e1",
                probs=c_probs,
            )
        with pred_r:
            render_prediction_card(
                title="Federated ResNet-50 (IID, FedAvg)",
                border_color="#68d391",
                accent="#68d391",
                probs=f_probs,
            )

        # Agreement / disagreement
        c_top = CLASS_NAMES[int(np.argmax(c_probs))]
        f_top = CLASS_NAMES[int(np.argmax(f_probs))]

        if c_top == f_top:
            st.markdown(f"""
                <div class='agree-box'>
                    ✅ Both models agree: <b>{CLASS_EMOJIS[c_top]} {c_top}</b>
                    &nbsp;·&nbsp; Centralized: {c_probs.max()*100:.1f}%
                    &nbsp;·&nbsp; Federated: {f_probs.max()*100:.1f}%
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='disagree-box'>
                    ⚠️ Models disagree —
                    Centralized: <b>{CLASS_EMOJIS[c_top]} {c_top}</b>
                    ({c_probs.max()*100:.1f}%) &nbsp;·&nbsp;
                    Federated: <b>{CLASS_EMOJIS[f_top]} {f_top}</b>
                    ({f_probs.max()*100:.1f}%)
                </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
        <p style='text-align:center; color:#4a5568; font-size:0.8rem;'>
            Cross-Domain Federated Transfer Learning for Satellite Image Classification
            Across Geographic Regions &nbsp;·&nbsp; ResNet-50 + FedAvg &nbsp;·&nbsp; EuroSAT RGB
        </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
