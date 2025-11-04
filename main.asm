# ==================================================================================================
# PROJECT  :   ̶S̶E̶K̶A̶I̶ Wiener Filter (M=5, N=10)
# FILE     :  main.asm
# --------------------------------------------------------------------------------------------------
# STRUCTURE:
# [0] Data Section
# [1] Main Routine
# [2] Core Function
# [3] Helper Function
# ==================================================================================================

.data

# --------------------------------------------------------------------------------------------------
# [PARAMETERS]
# --------------------------------------------------------------------------------------------------
N_word:    .word 10             # Number of samples (N)
M_word:    .word 7              # Filter length (M)

# --------------------------------------------------------------------------------------------------
# [SIGNALS]
# --------------------------------------------------------------------------------------------------
input_sig: .float 99.5, -1.2, 10.4, 1.2, 2434.5, 11.5, -0.5, 67.0, -1.5, 1.2

desired:
	.float 100.0, -1.2, 10.9, 1.1, 2435.3, 11.0, -0.8, 66.0, -0.9, 1.1

# --------------------------------------------------------------------------------------------------
# [WORKSPACE ARRAYS]
# --------------------------------------------------------------------------------------------------
gamma_xx:  .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0    # Autocorrelation results (length M)
gamma_dx:  .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # Cross-correlation (future use)

R_matrix: .space 196  # R_matrix: 49 float (MxM = 7x7) Toeplitz matrix

# [Cholesky Solver Buffers] ------------------------------------------------------------------------
L_matrix:      .space 196        # lower-triangular matrix L
y_vector:      .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

hopt:
	.float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

# Outputs
y_out:     .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
mmse_val:  .float 0.0

# --------------------------------------------------------------------------------------------------
# [STRINGS]
# --------------------------------------------------------------------------------------------------
lbl_hdr:      .asciiz "=== Wiener Filter Results ===\n"
lbl_outseq:   .asciiz "Output sequence (y):\n"
lbl_mmse:     .asciiz "\nMMSE: "

newline:
	.asciiz "\n"

# --------------------------------------------------------------------------------------------------
# [CONSTANTS]
# --------------------------------------------------------------------------------------------------
zero_float: .float 0.0
one_float:  .float 1.0

# =====================================================================================================================================
# [1] MAIN ROUTINE
# =====================================================================================================================================
.text

main:

# Load N and M into registers
la $t0, N_word
lw $s0, 0($t0)          # t1 = N
la $t1, M_word
lw $s1, 0($t1)          # t3 = M

# --- Call estimate_autocorrelation ---
la $a0, input_sig       # base of x[n]
la $a1, input_sig       # base of x[n]

move $a2, $s0             # N
move $a3, $s1             # maxlag

jal estimate_correlation

# --- Call estimate_crosscorrelation ---
la $a0, desired         # base of d[n]
la $a1, input_sig       # base of x[n]

move $a2, $s0             # N
move $a3, $s1             # maxlag

jal estimate_correlation

# Build R_matrix from gamma_xx
la   $a0, gamma_xx    # gamma_xx base
move $a1, $s1         # M
la   $a2, R_matrix           # R_matrix base (output)
jal  build_R_from_gamma_xx

# Solve R h = γd using Cholesky factorization
move $a0, $s1
jal  compute_hopt

# === DEBUGGING ============================================

# [DEBUG]: print gamma_xx array
la $a0, gamma_xx

move $a1, $s1             # length = M
jal  print_float_array

# [DEBUG]: print gamma_dx
la $a0, gamma_dx

move $a1, $s1
jal  print_float_array

# [DEBUG]: print R_matrix as M*M floats for debug:
la $a0, R_matrix

mul $a1, $s1, $s1    # a1 = M * M  (pseudo-instr; MARS supports this)
jal print_float_array

# [DEBUG]: print hopt matrix
la $a0, hopt

move $a1, $s1
jal  print_float_array

# === OWARI ============================================
# Exit
li $v0, 10
syscall

# =====================================================================================================================================
# = [2] CORE FUNCTIONS
# =====================================================================================================================================

# -------------------------------------------------------------------------------------
# FUNCTION: estimate_correlation
# -------------------------------------------------------------------------------------
# PURPOSE:
# Compute correlation sequence γ_xy[k] = (1/N) * Σ_n (x[n] * y[n−k]).
# Used for both autocorrelation (x=x) and cross-correlation (x≠y).
#
# INPUT:
# $a0 = base address of x[n]
# $a1 = base address of y[n]
# $a2 = N                # Signal length
# $a3 = maxlag (M)
#
# OUTPUT:
# Stores result in either gamma_xx (if x==y) or gamma_dx (if x!=y)
# -------------------------------------------------------------------------------------

estimate_correlation:
	addi $sp, $sp, -16          # Save registers
	sw   $ra, 12($sp)
	sw   $s0, 8($sp)
	sw   $s1, 4($sp)
	sw   $s2, 0($sp)

# If same address (x[n] and y[n])
# -> autocorrelation -> use gamma_xx else gamma_dx (cross-correlation)
bne $a0, $a1, not_same_addresses
la  $t8, gamma_xx
j   common_case

not_same_addresses:
	la $t8, gamma_dx

common_case:
	move $s0, $a0      # s0 = base of x (input signal)
	move $s1, $a1      # s1 = base of y (desired signal, or same as x for autocorrelation)
	move $s2, $a2      # s2 = N (signal length)
	move $s3, $a3      # s3 = M (max lag)

	li $t0, 0          # l = 0 (outer loop for lag)

correlation_outer_loop:
	bge $t0, $s3, correlation_done   # If l >= M, finish

# init $f0 = 0.0 (outer_loop accumulator)
mtc1 $zero, $f0

li $t1, 0          # n = 0 (inner loop)

correlation_inner_loop:
	bge $t1, $s2, correlation_sum_done  # If n >= N, finish this loop

# Calculate idx = n - l
sub $t2, $t1, $t0   # idx = n - l

# Skip invalid
bltz $t2, skip_product  # If idx < 0, skip product
bge  $t2, $s2, skip_product  # If idx >= N, skip product

# Load x[n] into f2
sll  $t3, $t1, 2       # byte offset = n * 4
add  $t3, $s0, $t3     # address of x[n]
lwc1 $f2, 0($t3)       # load x[n] into f2

# Load y[idx] into f4
sll  $t3, $t2, 2       # byte offset = idx * 4
add  $t3, $s1, $t3     # address of y[idx]
lwc1 $f4, 0($t3)       # load y[idx] into f4

# Multiply and accumulate: f0 += x[n] * y[idx]
mul.s $f6, $f2, $f4    # f6 = x[n] * y[idx]
add.s $f0, $f0, $f6    # sum += f6

skip_product:
	addi $t1, $t1, 1       # Increment n
	j    correlation_inner_loop

# Divide the sum by N
correlation_sum_done:
	mtc1    $s2, $f8       # move N into f8
	cvt.s.w $f8, $f8       # convert int to float
	div.s   $f10, $f0, $f8  # f10 = f0 / N

# Store result in gamma_xy[l] (store it into gamma_xx or gamma_dx based on usage)
sll  $t3, $t0, 2       # byte offset = l * 4
add  $t3, $t8, $t3    # address of gamma_xy[l]
swc1 $f10, 0($t3)     # store gamma_xy[l] into memory

# Next l
addi $t0, $t0, 1
j    correlation_outer_loop

correlation_done:
	lw   $ra, 12($sp)      # Restore registers
	lw   $s0, 8($sp)
	lw   $s1, 4($sp)
	lw   $s2, 0($sp)
	addi $sp, $sp, 16
	jr   $ra

# -------------------------------------------------------------------------------------
# FUNCTION: build_R_from_gamma_xx
# -------------------------------------------------------------------------------------
# PURPOSE:
# Build an M×M Toeplitz autocorrelation matrix R, where
# R[l, k] = γ_xx(|l − k|)
#
# INPUT:
# $a0 = base of γ_xx array (float[M])
# $a1 = M (matrix dimension)
# $a2 = base of R_matrix (float[M×M])  # output
#
# OUTPUT:
# Populates R_matrix[] in row-major order.
# -------------------------------------------------------------------------------------

build_R_from_gamma_xx:
	addi $sp, $sp, -16
	sw   $ra, 0($sp)
	sw   $s0, 4($sp)
	sw   $s1, 8($sp)
	sw   $s2, 12($sp)

	move $s0, $a0       # s0 = gamma_xx base
	move $s1, $a1       # s1 = M
	move $s2, $a2       # s2 = R_matrix base

	li $t0, 0         # l = 0

build_R_outer:
	bge $t0, $s1, build_R_done    # if l >= M done

	li $t1, 0         # k = 0

build_R_inner:
	bge $t1, $s1, build_R_next_l  # if k >= M -> next l

# compute lag = l - k
sub $t2, $t0, $t1
abs $t2, $t2

# if lag < gamma_len load gamma_xx[lag], else use 0.0
blt $t2, $s1, build_R_load_gamma

# load 0.0
mtc1 $zero, $f0
j    build_R_store

build_R_load_gamma:
	sll  $t3, $t2, 2        # byte offset = lag * 4
	add  $t3, $s0, $t3
	lwc1 $f0, 0($t3)

build_R_store:

# compute index = l * M + k
mul  $t3, $t0, $s1     # t7 = l * M
addu $t3, $t3, $t1    # t7 += k

sll $t3, $t3, 2      # byte offset = index * 4
add $t3, $s2, $t3

swc1 $f0, 0($t3)

addi $t1, $t1, 1
j    build_R_inner

build_R_next_l:
	addi $t0, $t0, 1
	j    build_R_outer

build_R_done:
	lw   $ra, 0($sp)
	lw   $s0, 4($sp)
	lw   $s1, 8($sp)
	lw   $s2, 12($sp)
	addi $sp, $sp, 16
	jr   $ra

# -------------------------------------------------------------------------------------
# FUNCTION: compute_hopt
# -------------------------------------------------------------------------------------
# PURPOSE:
# Solve the linear system R h = γ_d, where R is symmetric positive-definite.
# Implements the solution using Cholesky factorization and substitution:
#
# (1) Factorization:        R = L * L^T
# (2) Forward substitution: L * y = γ_d
# (3) Backward substitution: L^T * h = y
#
# This routine is a direct translation of the Python reference implementation
# `compute_hopt(R, b)` using Cholesky decomposition.
#
# INPUT:
# $a0 = n                  # Matrix dimension (M)
#
# GLOBALS USED:
# R_matrix   - Input correlation matrix (float[M][M])
# gamma_dx   - Right-hand vector γ_d (float[M])
# L_matrix   - Workspace for Cholesky lower triangle (float[M][M])
# y_vector   - Workspace for forward substitution (float[M])
# hopt       - Output vector h (float[M])
#
# OUTPUT:
# hopt[] = Solution vector for R * h = γ_d
#
# RETURN:
# None
# -------------------------------------------------------------------------------------

compute_hopt:
	addi $sp, $sp, -8
	sw   $ra, 4($sp)
	sw   $s0, 0($sp)

	move $s0, $a0 # n

# ------------------------------------------------------------------
# Step 1: Compute Cholesky factor L (R = L * L^T)
# ------------------------------------------------------------------
li $t0, 0 # i = 0

chol_outer_i:
	bge $t0, $s0, chol_done
	li  $t1, 0  # j = 0

chol_inner_j:
	bgt $t1, $t0, chol_next_i

# Compute sum s = Σ_{k=0}^{j-1} L[i, k] * L[j, k]
li   $t2, 0     # k = 0
mtc1 $zero, $f4 # s = 0

chol_sum_k:
	bge $t2, $t1, chol_sum_end

# load L[i, k] and L[j, k]
mul $t3, $t0, $s0
add $t3, $t3, $t2
sll $t3, $t3, 2
la  $t4, L_matrix
add $t3, $t3, $t4
l.s $f6, 0($t3)

mul $t3, $t1, $s0
add $t3, $t3, $t2
sll $t3, $t3, 2
add $t3, $t3, $t4
l.s $f8, 0($t3)

mul.s $f10, $f6, $f8
add.s $f4, $f4, $f10

addi $t2, $t2, 1
j    chol_sum_k

chol_sum_end:

# --- if i == j: diagonal element
bne $t0, $t1, chol_offdiag

# val = R[i, i] - s
mul $t3, $t0, $s0
add $t3, $t3, $t1
sll $t3, $t3, 2

la  $t4, R_matrix
add $t3, $t3, $t4 # R[i, j]

l.s $f12, 0($t3)

sub.s  $f12, $f12, $f4
sqrt.s $f12, $f12

# store L[i, j] = val
mul $t3, $t0, $s0
add $t3, $t3, $t1
sll $t3, $t3, 2
la  $t8, L_matrix
add $t3, $t3, $t8
s.s $f12, 0($t3)
j   chol_next_j

chol_offdiag:
# L[i, j] = (R[i, j] - s) / L[j, j]
mul $t3, $t0, $s0
add $t3, $t3, $t1
sll $t3, $t3, 2

la  $t4, R_matrix
add $t3, $t3, $t4
l.s $f14, 0($t3) # Value of R[i, j]

sub.s $f14, $f14, $f4

# load L[j, j]
mul $t3, $t1, $s0
add $t3, $t3, $t1
sll $t3, $t3, 2
la  $t4, L_matrix
add $t3, $t3, $t4
l.s $f16, 0($t3)

div.s $f14, $f14, $f16

# store L[i, j]
mul $t3, $t0, $s0
add $t3, $t3, $t1
sll $t3, $t3, 2
add $t3, $t3, $t4
s.s $f14, 0($t3)

chol_next_j:
	addi $t1, $t1, 1
	j    chol_inner_j

chol_next_i:
	addi $t0, $t0, 1
	j    chol_outer_i

chol_done:

# ------------------------------------------------------------------
# Step 2: Forward substitution  (L y = b)
# ------------------------------------------------------------------
li $t0, 0

forw_i:
	bge  $t0, $s0, back_subst
	mtc1 $zero, $f4
	li   $t1, 0

forw_k:
	bge $t1, $t0, forw_sum_done

# s += L[i, k] * y[k]
mul $t2, $t0, $s0
add $t2, $t2, $t1
sll $t2, $t2, 2
la  $t3, L_matrix
add $t2, $t2, $t3
l.s $f6, 0($t2)

sll $t2, $t1, 2

la  $t3, y_vector
add $t2, $t2, $t3
l.s $f8, 0($t2)

mul.s $f10, $f6, $f8
add.s $f4, $f4, $f10

addi $t1, $t1, 1
j    forw_k

forw_sum_done:
	sll $t2, $t0, 2
	la  $t3, gamma_dx
	add $t2, $t2, $t3
	l.s $f12, 0($t2)  # gamma_xx[i]

	sub.s $f12, $f12, $f4 # gamma_xx[i] - s

# divide by L[i, i]
mul $t3, $t0, $s0
add $t3, $t3, $t0

sll $t3, $t3, 2
la  $t4, L_matrix
add $t3, $t3, $t4
l.s $f14, 0($t3) # L[i, i]

div.s $f12, $f12, $f14

la  $t3, y_vector
sll $t4, $t0, 2
add $t3, $t3, $t4
s.s $f12, 0($t3)

addi $t0, $t0, 1
j    forw_i

# ------------------------------------------------------------------
# Step 3: Backward substitution  (L^T h = y)
# ------------------------------------------------------------------
back_subst:
	addi $t0, $s0, -1 # $t0 <- i

back_i:
	blt  $t0, $zero, chol_return
	mtc1 $zero, $f4

	addi $t1, $t0, 1  # $t1 <- k

back_k:
	bge $t1, $s0, back_sum_done

	mul $t2, $t1, $s0
	add $t2, $t2, $t0
	sll $t2, $t2, 2 # calculate inex of L[k, i]

	la  $t3, L_matrix
	add $t2, $t2, $t3
	l.s $f6, 0($t2) # Value of L[k, i]

	sll $t2, $t1, 2
	la  $t3, hopt
	add $t2, $t3, $t2
	l.s $f8, 0($t2) # Value of hopt[k]

	mul.s $f10, $f6, $f8
	add.s $f4, $f4, $f10 # s += L[k, i] * hopt[k]

	addi $t1, $t1, 1
	j    back_k

back_sum_done:
	sll $t2, $t0, 2
	la  $t3, y_vector
	add $t2, $t2, $t3
	l.s $f12, 0($t2) # Value of y[i]

	sub.s $f12, $f12, $f4 # y[i] - s

# divide by L[i, i]
mul $t2, $t0, $s0
add $t2, $t2, $t0
sll $t2, $t2, 2

la  $t3, L_matrix
add $t2, $t2, $t3
l.s $f14, 0($t2) # Value of L[i, i]

div.s $f12, $f12, $f14 # (y[i] - s) / L[i, i]

sll $t2, $t0, 2
la  $t3, hopt
add $t2, $t3, $t2 # address of x[i]

s.s $f12, 0($t2) # Load (y[i] - s) / L[i, i] --------> x[i]

addi $t0, $t0, -1
j    back_i

chol_return:
	lw   $ra, 4($sp)
	lw   $s0, 0($sp)
	addi $sp, $sp, 8
	jr   $ra

# =====================================================================================================================================
# === [HELPER]: I/O + DEBUG ===========================================================================================================
# === Erin Erin =======================================================================================================================
# =====================================================================================================================================

# ---------------------------------------------------------
# (help-func) print_float_array

# Inputs:
# a0 = base address of float array
# a1 = length (int)
# ---------------------------------------------------------
print_float_array:
	addi $sp, $sp, -8
	sw   $ra, 4($sp)
	sw   $s0, 0($sp)

	move $s0, $a0     # s0 = base address
	li   $t0, 0        # index = 0

.print_loop:
	bge $t0, $a1, .print_done

# load float element
sll  $t1, $t0, 2
add  $t2, $s0, $t1
lwc1 $f12, 0($t2)

# syscall 2 = print float
li $v0, 2
syscall

# print a space
li $v0, 11
li $a0, 32      # ASCII for ' '
syscall

addi $t0, $t0, 1
j    .print_loop

.print_done:
# print newline
li $v0, 4
la $a0, newline
syscall

lw   $ra, 4($sp)
lw   $s0, 0($sp)
addi $sp, $sp, 8
jr   $ra
