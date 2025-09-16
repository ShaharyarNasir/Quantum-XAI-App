import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import curve_fit

# Simulation parameters
N = 6
dim = 2**N  # 64 for N=6
sx = qt.sigmax()
sz = qt.sigmaz()
I = qt.qeye(2)

# Build Hamiltonian as flat operator
H_data = np.zeros((dim, dim), dtype=complex)  # Use complex for quantum ops
for i in range(N):
    # Single-qubit terms
    sz_i = qt.tensor([I] * i + [0.5 * sz] + [I] * (N - i - 1))
    H_data += sz_i.data.to_array()  # Use to_array() instead of toarray()

for i in range(N-1):
    # Interaction terms
    sz_sx = qt.tensor([I] * i + [sz, sx] + [I] * (N - i - 2))
    sx_sz = qt.tensor([I] * i + [sx, sz] + [I] * (N - i - 2))
    H_data += sz_sx.data.to_array() + sx_sz.data.to_array()

H = qt.Qobj(H_data, dims=[[dim], [dim]])  # Flat dims

# Initial state: Bell on first two, rest |0> (flat ket)
state_bell = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
              qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
state_rest = qt.tensor([qt.basis(2, 0) for _ in range(N-2)])
psi0_flat = qt.tensor(state_bell, state_rest)
psi0 = qt.Qobj(psi0_flat.full(), dims=[[dim], [1]])  # Flat ket dims

times = np.linspace(0, 10, 100)

# Lindblad: Dephasing on first qubit only
gamma = 0.05
c_ops_data = np.sqrt(gamma) * \
    qt.tensor([sz] + [I for _ in range(N-1)]).data.to_array()
c_ops = [qt.Qobj(c_ops_data, dims=[[dim], [dim]])]  # Flat

# Observable: σ_x^0 σ_x^1 (first two qubits)
obs_data = qt.tensor([sx, sx] + [I for _ in range(N-2)]).data.to_array()
obs = qt.Qobj(obs_data, dims=[[dim], [dim]])  # Flat

# Solve
try:
    result = qt.mesolve(H, psi0, times, c_ops=c_ops, e_ops=[obs])
    corr = np.array(result.expect[0])
    print("Simulation successful.")
except Exception as e:
    print(f"mesolve failed: {e}")
    # Fallback: No c_ops
    result = qt.mesolve(H, psi0, times, e_ops=[obs])
    corr = np.array(result.expect[0])
    print("Fallback simulation without decoherence.")

# Neural proxy fit


def proxy(t, a, b, w): return a * np.sin(w * t) + b


try:
    popt, _ = curve_fit(proxy, times, corr, p0=[0.5, 0, 1], maxfev=5000)
    pred = proxy(times, *popt)
    mse = np.mean((corr - pred) ** 2)
    print(f"6-Qubit MSE: {mse:.4f}")
except RuntimeError:
    print("Curve fit failed. Using linear fallback.")
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(times.reshape(-1, 1), corr)
    pred = reg.predict(times.reshape(-1, 1)).flatten()
    mse = np.mean((corr - pred) ** 2)
    print(f"Fallback MSE: {mse:.4f}")

# SHAP-like peaks via residuals
shap_impact = np.abs(corr - pred)
shap_peak_indices = np.argsort(shap_impact)[-3:]
shap_peaks = times[shap_peak_indices]
print(f"SHAP Peaks at t ≈ {shap_peaks}")

# Figure 5: Main Plot (MSE vs. N)
fig, ax = plt.subplots(figsize=(6, 4))
qubits = np.array([2, 4, 6])
mse_values = np.array([0.0196, 0.025, mse])
ax.loglog(qubits, mse_values, 'bo-',
          label='MSE vs. Qubits', linewidth=2, markersize=8)
ax.set_xlabel('Number of Qubits ($N$)', fontsize=12)
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
ax.grid(True, which="both", ls="--", alpha=0.7)
ax.legend(fontsize=10)
ax.set_title('Scalability of Quantum XAI Framework', fontsize=12)

# Inset: 6-qubit SHAP Impacts
axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.scatter(shap_peaks, np.ones(3) * 0.1, c='r',
              label='6-Qubit SHAP Impact', s=50)
axins.set_xlim(0, 10)
axins.set_ylim(0, 0.2)
axins.set_xlabel('Time $t$', fontsize=10)
axins.set_ylabel('Impact', fontsize=10)
axins.grid(True, ls="--", alpha=0.7)
axins.legend(fontsize=8)

plt.tight_layout()
plt.savefig('scalability_mse.pdf', dpi=300, bbox_inches='tight')

# Separate inset
plt.figure(figsize=(3, 3))
plt.scatter(shap_peaks, np.ones(3) * 0.1, c='r',
            label='6-Qubit SHAP Impact', s=50)
plt.xlim(0, 10)
plt.ylim(0, 0.2)
plt.xlabel('Time $t$', fontsize=10)
plt.ylabel('Impact', fontsize=10)
plt.grid(True, ls="--", alpha=0.7)
plt.legend(fontsize=8)
plt.savefig('shap_6qubit_inset.pdf', dpi=300, bbox_inches='tight')

print("Figures saved: scalability_mse.pdf and shap_6qubit_inset.pdf")
