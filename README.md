# Wiener Filter (MIPS Assembly)
## 🧮 Mathematical Background

The **Wiener filter** minimizes the **mean squared error (MSE)** between the desired signal ( d(n) ) and the filter output ( y(n) ):

[
J = E\left[(d(n) - y(n))^2\right]
]

For a **finite impulse response (FIR)** filter of length ( M ):

[
y(n) = \sum_{k=0}^{M-1} h_k , x(n - k)
]

We find the **optimal coefficients** ( h = [h_0, h_1, \ldots, h_{M-1}]^T ) that minimize ( J ) using the **Wiener–Hopf equations**:

[
R , h_{\text{opt}} = \gamma_{dx}
]

where:

* ( R ) — autocorrelation (Toeplitz) matrix of input ( x(n) ):
  [
  R_{l,k} = \gamma_{xx}(l - k)
  ]
* ( \gamma_{xx}(m) ) — autocorrelation function:
  [
  \gamma_{xx}(m) = E[x(n) , x(n - m)]
  ]
* ( \gamma_{dx}(l) ) — cross-correlation vector between ( d(n) ) and ( x(n) ):
  [
  \gamma_{dx}(l) = E[d(n) , x(n - l)]
  ]

Once ( h_{\text{opt}} ) is obtained, the **filtered output** is:

[
y(n) = \sum_{k=0}^{M-1} h_{\text{opt}}[k] , x(n - k)
]

and the **minimum mean-square error (MMSE)** is computed as:

[
\text{MMSE} = \frac{1}{N} \sum_{n=0}^{N-1} (d(n) - y(n))^2
]

---

## 🧩 Computational Steps

1. **Estimate autocorrelation** ( \gamma_{xx}(k) )
   Using:
   [
   \gamma_{xx}(k) = \frac{1}{N} \sum_{n=0}^{N-1} x(n) , x(n-k)
   ]

2. **Estimate cross-correlation** ( \gamma_{dx}(l) )
   Using:
   [
   \gamma_{dx}(l) = \frac{1}{N} \sum_{n=0}^{N-1} d(n) , x(n-l)
   ]

3. **Form Toeplitz matrix** ( R ) from ( \gamma_{xx} )

4. **Solve** ( R h = \gamma_{dx} )
   Using **Cholesky decomposition**:
   [
   R = L L^T, \quad
   L y = \gamma_{dx}, \quad
   L^T h = y
   ]

5. **Filter** input ( x(n) ) using ( h )
   and compute the MMSE.
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
