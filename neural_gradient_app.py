import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Neural Gradient Pattern Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E8F1F5 0%, #D4E6F1 100%);
        border-right: 2px solid #AED6F1;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #184F72 !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background-color: #F8F9F9;
        border: 1px solid #D5DBDB;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Neural Gradient Pattern Predictor")
st.subheader("Custom NumPy Neural Network, Loss Convergence & Gradient Descent Lab")
st.markdown("---")

st.sidebar.header("🎛️ Neural Training Controls")
learning_rate = st.sidebar.slider("Learning Rate (Step Size)", 0.001, 0.050, 0.015, 0.001)
epochs = st.sidebar.slider("Training Epochs (Iterations)", 100, 1500, 600, 50)
hidden_neurons = st.sidebar.slider("Hidden Layer Neurons", 4, 32, 12, 2)
noise_level = st.sidebar.slider("Signal Noise Level", 0.00, 0.25, 0.06, 0.01)

try:
    np.random.seed(13)

    x = np.linspace(-1.0, 1.0, 120).reshape(-1, 1)
    true_signal = 0.65 * np.sin(np.pi * x) + 0.25 * (x ** 2)
    noise = np.random.normal(0.0, noise_level, size=x.shape)
    y = true_signal + noise

    n_samples = x.shape[0]

    W1 = np.random.randn(1, hidden_neurons) * np.sqrt(2.0 / (1 + hidden_neurons))
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons, 1) * np.sqrt(2.0 / (hidden_neurons + 1))
    b2 = np.zeros((1, 1))

    loss_history = []
    initial_loss = None
    final_loss = None
    y_pred = None

    for epoch in range(1, epochs + 1):

        z1 = x @ W1 + b1
        a1 = np.tanh(z1)
        y_pred = a1 @ W2 + b2

        error = y_pred - y
        mse_loss = float(np.mean(error ** 2))

        if epoch == 1:
            initial_loss = mse_loss
        if epoch == epochs:
            final_loss = mse_loss

        loss_history.append({
            "Epoch": epoch,
            "MSE_Loss": round(mse_loss, 6)
        })

        d_loss_d_ypred = (2.0 / n_samples) * error
        dW2 = a1.T @ d_loss_d_ypred
        db2 = np.sum(d_loss_d_ypred, axis=0, keepdims=True)

        d_loss_d_a1 = d_loss_d_ypred @ W2.T
        d_loss_d_z1 = d_loss_d_a1 * (1.0 - np.tanh(z1) ** 2)
        dW1 = x.T @ d_loss_d_z1
        db1 = np.sum(d_loss_d_z1, axis=0, keepdims=True)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    loss_df = pd.DataFrame(loss_history)
    loss_reduction_pct = (
        ((initial_loss - final_loss) / initial_loss) * 100.0
        if initial_loss and initial_loss > 0 else 0.0
    )

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_like_score = max(0.0, min(100.0, (1.0 - (ss_res / ss_tot)) * 100.0)) if ss_tot > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="📉 Final MSE Loss",
            value=f"{final_loss:.5f}",
            delta=f"Epochs: {epochs}"
        )
    with col2:
        st.metric(
            label="🚀 Initial MSE Loss",
            value=f"{initial_loss:.5f}"
        )
    with col3:
        st.metric(
            label="🎯 Loss Reduction",
            value=f"{loss_reduction_pct:.1f}%"
        )
    with col4:
        st.metric(
            label="📐 Pattern Fit Score",
            value=f"{r2_like_score:.1f}%"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=loss_df["Epoch"],
            y=loss_df["MSE_Loss"],
            mode="lines",
            name="Training MSE Loss",
            line=dict(color="#C0392B", width=2.5)
        ))
        fig_loss.update_layout(
            title="Gradient Descent Loss Convergence",
            xaxis_title="Epoch",
            yaxis_title="Mean Squared Error (MSE)",
            template="plotly_white",
            hovermode="x unified",
            height=380
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with chart_col2:
        order = np.argsort(x[:, 0])
        x_sorted = x[order, 0]
        y_sorted = y[order, 0]
        pred_sorted = y_pred[order, 0]

        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=x_sorted,
            y=y_sorted,
            mode="markers",
            name="Observed Data",
            marker=dict(color="#2E86C1", size=6, opacity=0.75)
        ))
        fig_fit.add_trace(go.Scatter(
            x=x_sorted,
            y=pred_sorted,
            mode="lines",
            name="Neural Network Prediction",
            line=dict(color="#1E8449", width=3)
        ))
        fig_fit.update_layout(
            title="Actual Pattern vs. Neural Prediction",
            xaxis_title="Input Feature (x)",
            yaxis_title="Target Output (y)",
            template="plotly_white",
            hovermode="x unified",
            height=380
        )
        st.plotly_chart(fig_fit, use_container_width=True)

    with st.expander("📋 View Epoch-by-Epoch Training Log"):
        st.dataframe(loss_df, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Neural Engine Error: {e}")
    st.info("Adjust learning rate, epochs, or hidden neurons and try again.")
