# Wiener Filter (MIPS Assembly)
## 🧮 Mathematical Background

The Wiener filter minimizes the mean-square error between desired signal \( d(n) \) and filtered output \( y(n) \).

<img src="https://latex.codecogs.com/gif.latex?y(n)=\sum_{k=0}^{M-1}h_kx(n-k)" />

<img src="https://latex.codecogs.com/gif.latex?E_M=E\left[\left|d(n)-\sum_{k=0}^{M-1}h_kx(n-k)\right|^2\right]" />

The optimal coefficients \( h_{opt} \) satisfy:

<img src="https://latex.codecogs.com/gif.latex?\sum_{k=0}^{M-1}h_k\gamma_{xx}(l-k)=\gamma_{dx}(l)" />

Matrix form:

<img src="https://latex.codecogs.com/gif.latex?R_Mh_M=\gamma_d" />

Solution:

<img src="https://latex.codecogs.com/gif.latex?h_{opt}=R_M^{-1}\gamma_d" />

Minimum Mean-Square Error:

<img src="https://latex.codecogs.com/gif.latex?MMSE=\sigma_d^2-\gamma_d^T R_M^{-1}\gamma_d" />

---

## 📂 Project Structure

```

main.asm          # MIPS implementation
reference.py      # Reference Python model
README.md         # Documentation
desired.txt       # Desired sequence
input.txt         # Noisy input sequence
output.txt        # Output (results)

```

---

## 🧰 Implemented Functions

| Function | Purpose |
|-----------|----------|
| `estimate_correlation` | Compute auto & cross correlation |
| `build_R_from_gamma_xx` | Build Toeplitz autocorrelation matrix |
| `compute_hopt` | Solve \( R h = \gamma_d \) using Cholesky decomposition |
| `filter_signal` | Apply FIR filter to input |
| `compute_mmse` | Compute mean squared error |

---

## 🧱 TODOs

### Logic
- [ ] Implement file reading (`input.txt`, `desired.txt`)
- [ ] Implement file writing (`output.txt`)

### Printing/Log
- [ ] Add input size validation
- [ ] Clean up debug print

### Report
- [ ] Python Script for testing large scale
- [ ] Include screenshots & report diagrams

---

## 🧪 Testing

Run both:
- **Python**: chưa có
- **MARS** (`main.asm`) and compare printed MMSE & output sequence.

---

## 🏁 Expected Output Format

```

<space-separated filtered output>
<MMSE value>
```

Example:

```
99.6 -1.1 10.8 1.0 2435.0 10.9 -0.7 66.2 -1.0 1.1
0.002310
```

![troll](https://scontent.fsgn5-15.fna.fbcdn.net/v/t39.30808-1/541528486_3987213001590708_6150294427196741489_n.jpg?stp=dst-jpg_s200x200_tt6&_nc_cat=111&ccb=1-7&_nc_sid=e99d92&_nc_eui2=AeF803EYzXWATKcC13o1mIIqEtM4bEUR298S0zhsRRHb3w1CIA8AzPJiUXTANHUY3JA1zC1Qk7ya_F5hG6ZdsB6m&_nc_ohc=PjgNXG1JA2sQ7kNvwG29Wkk&_nc_oc=AdntknupfpvLpDAJJrmiKFyQYb1KSPQwvITikwGoiKwmQIE4xIzXrwd9KigVdvnuD2548EAnijSPWd1Cbe7yfxjg&_nc_zt=24&_nc_ht=scontent.fsgn5-15.fna&_nc_gid=ONPQy2FxgsQBm2VuBL8Zrw&oh=00_AfiJFDt-q-C3Mhveu5DaUqi_K5NC_Zx1UiG8MQcigveydQ&oe=690FAF3D)
