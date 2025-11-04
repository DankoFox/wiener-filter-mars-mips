# Wiener Filter (MIPS Assembly)
## 🧮 Mathematical Background

The Wiener filter minimizes the **mean-square error (MSE)** between the desired signal \(d(n)\) and the filter output \(y(n)\):  

```math
J = E\!\left[(\,d(n) - y(n)\,)^{2}\right]
```

For a finite-impulse-response (FIR) filter of length \(M\):  
```math
y(n) = \sum_{k=0}^{M-1} h_{k}\;x(n - k)
```

We find the **optimal coefficients** $\(h = [\,h_{0}, h_{1}, \dots, h_{M-1}\,]^{T}\)$ that minimise $\(J\)$ using the Wiener–Hopf equations:  
```math
R\,h_{\text{opt}} = \gamma_{dx}
```

where:

R = the autocorrelation (Toeplitz) matrix of the input (x(n)):

  ```math
  R_{l,k} = \gamma_{xx}(l - k)
  ```

$\gamma_{xx}(m)$ = autocorrelation function:

  ```math
  \gamma_{xx}(m) = E\!\bigl[x(n)\;x(n - m)\bigr]
  ```

$\gamma_{dx}(l)$ = cross-correlation vector between (d(n)) and (x(n)):

  ```math
  \gamma_{dx}(l) = E\!\bigl[d(n)\;x(n - l)\bigr]
  ```

Once $h_{\text{opt}}$ is obtained, the filtered output is:

```math
y(n) = \sum_{k=0}^{M-1} h_{\text{opt}}[k]\;x(n - k)
```

and the minimum mean-square error (MMSE) is computed as:

```math
\text{MMSE} = \frac{1}{N} \sum_{n=0}^{N-1} \bigl(d(n) - y(n)\bigr)^{2}
```

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
- [ ] Python Script for testing large scale (possible only after implement reading + reading)
- [ ] Include screenshots & report diagrams

---

## 🧪 Testing

Run both:
- **Python Script**: chưa có
- **MARS** (`main.asm`) and compare printed MMSE & output sequence.

---

## 🏁 Expected Output Format (MARS)

```

<space-separated filtered output>
<MMSE value>
```

Example:

```
99.6 -1.1 10.8 1.0 2435.0 10.9 -0.7 66.2 -1.0 1.1
0.002310
```

![troll](https://scontent.fsgn5-5.fna.fbcdn.net/v/t39.30808-6/481148404_2120531655076172_5403178123874385227_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=a5f93a&_nc_eui2=AeHMX-orpYUezFYxD1iRGjDXxwb7GxJcvBLHBvsbEly8Ei2RpeqjDpX2qZS3nW2ujiJxBGCv20w45tjy5mflDzxh&_nc_ohc=0Oj1qjsKQqEQ7kNvwFsHAQP&_nc_oc=AdnxmDkv8rtRbVBorgYXPrZrUjV1e3WR7y-3jNTo7UxXFaoz53dmdznENdBZ_9XwIKcfsCYhgqKxPzDR_eFpiLBC&_nc_zt=23&_nc_ht=scontent.fsgn5-5.fna&_nc_gid=gnLe6CMk_VcZy9aMJ-8ukA&oh=00_AfjrlLT_qtIVpgHLGeu13dxFcj17yJGvmzWLHQpJvpn4sg&oe=690FBA6C)
