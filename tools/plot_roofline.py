import numpy as np
import matplotlib.pyplot as plt

# Constants
peak_flops = 7680  # GFLOPS
theoretical_bw = 320  # GB/s
actual_bw = 220  # GB/s

# For n=2048 matrix multiplication
n = 2048
flops = 2 * n**3
bytes = (2 * n**3 / (16 * 8)) + n**2  # load + store
arithmetic_intensity = flops / bytes

print(f"Arithmetic intensity: {arithmetic_intensity:.2f} FLOPS/byte")

# Achieved performance
achieved_performance = 4779.9  # GFLOPS

# Generate x values (arithmetic intensity)
x = np.linspace(0, 1000, 1000)

# Calculate roofline values
theoretical_roof = np.minimum(peak_flops, theoretical_bw * x)
actual_roof = np.minimum(peak_flops, actual_bw * x)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot rooflines
plt.loglog(x, theoretical_roof, "b-", label=f"Theoretical BW ({theoretical_bw} GB/s)")

theoretical_q = peak_flops / theoretical_bw
plt.axvline(x=theoretical_q, color="b", linestyle=":", alpha=0.5)

plt.text(
    theoretical_q * 1.1,
    1000,
    f"q = {theoretical_q:.1f} Flops/byte",
    rotation=90,
    color="b",
)


# Plot achieved performance point
lines = plt.plot(
    arithmetic_intensity, achieved_performance, "ro", label="Achieved Performance"
)

plt.xlabel("Arithmetic Intensity (FLOPS/byte)")
plt.ylabel("Performance (GFLOPS/s)")

plt.title("Roofline Model: Achieved Performance")
plt.legend()

plt.savefig("roofline_2048_perf.png")

for line in lines:
    line.remove()


plt.loglog(x, actual_roof, "g-", label=f"Actual BW ({actual_bw} GB/s)")

actual_q = peak_flops / actual_bw
plt.axvline(x=actual_q, color="g", linestyle=":", alpha=0.5)
plt.text(actual_q * 1.1, 1000, f"q = {actual_q:.1f} Flops/byte", rotation=90, color="g")


plt.title("Roofline Model: Theoretical vs Actual Memory Bandwidth")
plt.legend()

plt.savefig("roofline_theoretical_vs_actual.png")
