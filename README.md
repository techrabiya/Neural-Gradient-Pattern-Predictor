🧠 NEURAL GRADIENT PATTERN PREDICTOR

An interactive two-layer neural network and gradient descent laboratory built from scratch in Python, NumPy, Pandas, Plotly, and Streamlit to visualize how models learn patterns and reduce error step by step.

🌐 System Overview

The Neural Gradient Pattern Predictor trains a custom neural network directly using NumPy matrix operations, forward propagation, tanh hidden activation, and Gradient Descent weight updates—without hiding the core math inside a black-box ML library. Users can adjust the Learning Rate, Training Epochs, Hidden Layer Neurons, and Signal Noise Level to observe how training error (MSE Loss) converges and how closely the predicted curve fits the underlying pattern.

🛠️ Architecture & Tech Stack

1. Core Engine: Python (Forward pass, backpropagation gradients, parameter updates, and fit evaluation)
2. Matrix & Learning Math: NumPy (Weight/bias initialization, matrix multiplications, derivatives, and Mean Squared Error loss)
3. Tabular & Training Logs: Pandas (Epoch-by-epoch loss tracking and structured training tables)
4. Interactive Visualization: Plotly (plotly.graph_objects side-by-side loss convergence chart and pattern-fit comparison chart)
5. Web Dashboard & Controls: Streamlit (Sidebar training controls, 4 KPI metric cards, dual charts, and expandable epoch log)

🎛️ Interactive Controls & Metrics

Sidebar Controls:

1. Learning Rate (Step Size) (0.001 to 0.050)
2. Training Epochs (Iterations) (100 to 1500)
3. Hidden Layer Neurons (4 to 32)
4. Signal Noise Level (0.00 to 0.25)

Top KPI Cards:
1. Final MSE Loss (with active epoch count)
2. Initial MSE Loss
3. Loss Reduction (%)
4. Pattern Fit Score (%)

📁 Project Structure

1. neural_gradient_app.py — Main Streamlit application containing the custom neural network, gradient descent loop, KPI cards, Plotly charts, and epoch log
2. requirements.txt — Python dependencies (streamlit, numpy, pandas, plotly)
3. README.md — Project documentation, setup guide, feedback section, and community links

🚀 Quick Start & Local Setup

Run the project locally in your terminal:

1. Clone the repository:
   git clone https://github.com/techrabiya/Neural-Gradient-Pattern-Predictor
2. Enter the project folder: cd Neural-Gradient-Pattern-Predictor
3. Install dependencies: pip install -r requirements.txt
4. Launch the app:https://neural-gradient-pattern-predictor-apqptpnmxt4najkd3gwnoj.streamlit.app/

💬 Comments, Feedback & Discussion

I love learning from builders, engineers, students, and data enthusiasts! If you test this app or read the code, share your thoughts in the Issues / Discussions / LinkedIn post comments:

1. What happens when you push Learning Rate too high vs. keeping it low?
2. Which activation function or architecture improvement would you test next on this build?
3. Spot a way to make the math or visualization even cleaner? Drop your feedback—every tip helps me grow!

🔔 Stay Connected & Subscribe

1. GitHub Profile: Follow @techrabiya for new AI, Python, data, and Streamlit builds
2. LinkedIn: Connect & follow for live project demos, walkthroughs, and learning updates
3. YouTube / Video Demos: Subscribe to my channel for project breakdowns and Excel/data walkthroughs

👑 Built By
  Rabia Razzaq Khan — Self-Taught Developer & AI/Data Builder
