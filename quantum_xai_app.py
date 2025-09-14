import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qutip import basis, tensor, sigmaz, sigmax, mesolve, qeye
import shap
import matplotlib.pyplot as plt
import gradio as gr
import io
import base64  # needed for SHAP plot encoding

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)


def simulate_and_train(num_qubits=2, max_time=10, num_points=100, epochs=1000, decoherence_rate=0.0):
    """
    Simulate entangled qubits with optional decoherence,
    train a small neural network proxy, and compute SHAP values.
    """

    # Hamiltonian for n qubits (linear chain) with magnetic field
    H_terms = []
    for i in range(num_qubits - 1):
        ops = [qeye(2)] * num_qubits
        ops[i] = sigmaz()
        ops[i + 1] = sigmax()
        H_terms.append(tensor(ops))

        ops = [qeye(2)] * num_qubits
        ops[i] = sigmax()
        ops[i + 1] = sigmaz()
        H_terms.append(tensor(ops))

    # Magnetic field term on first qubit
    H_field = 0.5 * tensor(sigmaz(), qeye(2)) if num_qubits == 2 else sum(
        0.5 * tensor(sigmaz() if i == 0 else qeye(2),
                     *[qeye(2)] * (num_qubits - 1))
        for i in range(num_qubits)
    )
    H = sum(H_terms) + H_field

    # Initial state: superposition on first qubit
    superpos = (tensor(basis(2, 0), basis(2, 0)) +
                tensor(basis(2, 1), basis(2, 0))).unit()
    psi0 = superpos if num_qubits == 2 else tensor(
        [basis(2, 0) for _ in range(num_qubits)])

    # Observable: σ_x ⊗ σ_x (or endpoints for larger)
    obs = tensor(sigmax(), sigmax()) if num_qubits == 2 else tensor(
        [sigmax() if i in (0, num_qubits - 1) else qeye(2)
         for i in range(num_qubits)]
    )

    times = np.linspace(0, max_time, num_points)

    # Optional decoherence: bit-flip channel
    c_ops = []
    if decoherence_rate > 0:
        for i in range(num_qubits):
            ops = [qeye(2)] * num_qubits
            ops[i] = sigmax()
            c_ops.append(np.sqrt(decoherence_rate) * tensor(ops))

    result = mesolve(H, psi0, times, c_ops, [obs])
    quantum_data = np.array(result.expect[0])
    inputs = times.reshape(-1, 1)
    outputs = quantum_data.reshape(-1, 1)

    # Neural network (scaled with qubit count)
    hidden1 = max(20, 10 * num_qubits)
    hidden2 = max(10, 5 * num_qubits)

    class QuantumNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.fc3 = nn.Linear(hidden2, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = QuantumNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(inputs_tensor)
        loss = loss_fn(preds, outputs_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # SHAP explanation on subset
    subset_size = min(20, num_points)
    explainer = shap.DeepExplainer(model, inputs_tensor[:subset_size])
    shap_values = explainer.shap_values(
        inputs_tensor[:subset_size], check_additivity=False)

    shap_fig = plt.figure()
    shap.summary_plot(shap_values, inputs[:subset_size], feature_names=[
                      "Time"], show=False)
    shap_buf = io.BytesIO()
    shap_fig.savefig(shap_buf, format='png')
    shap_base64 = base64.b64encode(shap_buf.getvalue()).decode('utf-8')
    plt.close(shap_fig)

    # Predictions
    with torch.no_grad():
        predicted = model(inputs_tensor).numpy()

    # Visualization: 2x2 dashboard
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curve
    axs[0, 0].plot(range(epochs), losses)
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MSE")

    # Prediction vs true
    axs[0, 1].plot(times, quantum_data, label="True Correlation")
    axs[0, 1].plot(times, predicted, '--', label="NN Prediction")
    axs[0, 1].set_title("Quantum Correlation vs. NN Prediction")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Correlation")
    axs[0, 1].legend()

    # SHAP plot
    shap_img = io.BytesIO(base64.b64decode(shap_base64))
    axs[1, 0].imshow(plt.imread(shap_img))
    axs[1, 0].set_title("SHAP: Time Impact on Predictions")
    axs[1, 0].axis('off')

    # Summary text
    final_mse = loss_fn(preds, outputs_tensor).item()
    axs[1, 1].text(0.5, 0.5,
                   f"Final MSE: {final_mse:.4f}\nQubits: {num_qubits}\nDecoherence: {decoherence_rate}",
                   ha='center', va='center', transform=axs[1, 1].transAxes)
    axs[1, 1].set_title("Summary")
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig


# Gradio Interface
def run_simulation(num_qubits, max_time, num_points, epochs, decoherence_rate):
    fig = simulate_and_train(num_qubits, max_time,
                             num_points, epochs, decoherence_rate)
    return fig


with gr.Blocks(title="Quantum XAI Mini-App") as demo:
    gr.Markdown(
        "# Quantum XAI Simulator\n"
        "Explore entangled qubit dynamics with a neural network proxy and SHAP interpretability.\n"
        "Adjust parameters, run simulations, and visualize results interactively."
    )

    with gr.Row():
        num_qubits = gr.Slider(2, 4, value=2, label="Number of Qubits", step=1)
        max_time = gr.Slider(5, 15, value=10, label="Max Time")
        num_points = gr.Slider(50, 200, value=100, label="Time Points")
        epochs = gr.Slider(500, 2000, value=1000, label="Training Epochs")
        decoherence_rate = gr.Slider(
            0.0, 0.1, value=0.0, label="Decoherence Rate")

    run_btn = gr.Button("Run Simulation")
    output_plot = gr.Plot(label="Results")

    run_btn.click(
        fn=run_simulation,
        inputs=[num_qubits, max_time, num_points, epochs, decoherence_rate],
        outputs=output_plot,
    )

if __name__ == "__main__":
    demo.launch(share=False)
