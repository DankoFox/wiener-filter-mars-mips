# ==================================================================================================
# PROJECT  :   ̶S̶E̶K̶A̶I̶ Wiener Filter (M=7, N=10)
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
input_signal:   .space 40
desired_signal: .space 40

# --------------------------------------------------------------------------------------------------
# [WORKSPACE ARRAYS]
# --------------------------------------------------------------------------------------------------
gamma_xx:  .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0    # Autocorrelation results (length M)
gamma_dx:  .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # Cross-correlation (future use)

R_matrix: .space 196  # R_matrix: 49 float (MxM = 7x7) Toeplitz matrix

# [Cholesky Solver Buffers] ------------------------------------------------------------------------
L_matrix:      .space 196        # lower-triangular matrix L
temp_vector:      .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

optimize_coefficient:
	.float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

# Outputs
output_signal:     .float 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

mmse:
	.float 0.0

# --------------------------------------------------------------------------------------------------
# [STRINGS]
# --------------------------------------------------------------------------------------------------
lbl_hdr:      .asciiz "=== Wiener Filter Results ===\n"
lbl_outseq:   .asciiz "Output sequence (y):\n"
lbl_filteredoutput: .asciiz "Filtered output: "
lbl_mmse:   .asciiz "\nMMSE: "
space_str:  .asciiz " "
input_size_error: .asciiz "Error: size not match\n"

newline:    .asciiz "\n"

# --------------------------------------------------------------------------------------------------
# [I/O]
# --------------------------------------------------------------------------------------------------
input_file: .asciiz "input.txt"
desired_file: .asciiz "desired.txt"
output_file: .asciiz "output.txt"

# --------------------------------------------------------------------------------------------------
# [CONSTANTS]
# --------------------------------------------------------------------------------------------------
zero_float: .float 0.0
ten_float: .float 10.0 # i hate you i hate you i hate you i hate you kkk

read_buf:   .space 1024 # buffer for input file
num_buf:    .space 32   # buffer for output file

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

jal read_input_and_desired_from_files

# --- Call estimate_autocorrelation ---
la $a0, input_signal       # base of x[n]
la $a1, input_signal       # base of x[n]

move $a2, $s0             # N
move $a3, $s1             # maxlag

jal estimate_correlation

# --- Call estimate_crosscorrelation ---
la $a0, desired_signal         # base of d[n]
la $a1, input_signal       # base of x[n]

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

# Filter input using hopt
move $a0, $s0
move $a1, $s1
jal  filter_signal

# Calculate MMSE
move $a0, $s0
jal  compute_mmse

jal write_output_signal_to_file

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
la $a0, optimize_coefficient

move $a1, $s1
jal  print_float_array

# [DEBUG]: filtered output
la $a0, output_signal

move $a1, $s0
jal  print_float_array

# [DEBUG]: MMSE
la $a0, mmse

li  $a1, 1
jal print_float_array

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
# temp_vector - Workspace for forward substitution (float[M])
# optimize_coefficient       - Output vector h (float[M])
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

# s += L[i, k] * temp[k]
mul $t2, $t0, $s0
add $t2, $t2, $t1
sll $t2, $t2, 2
la  $t3, L_matrix
add $t2, $t2, $t3
l.s $f6, 0($t2)

sll $t2, $t1, 2

la  $t3, temp_vector
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

la  $t3, temp_vector
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
	la  $t3, optimize_coefficient
	add $t2, $t3, $t2
	l.s $f8, 0($t2) # Value of hopt[k]

	mul.s $f10, $f6, $f8
	add.s $f4, $f4, $f10 # s += L[k, i] * hopt[k]

	addi $t1, $t1, 1
	j    back_k

back_sum_done:
	sll $t2, $t0, 2
	la  $t3, temp_vector
	add $t2, $t2, $t3
	l.s $f12, 0($t2) # Value of temp[i]

	sub.s $f12, $f12, $f4 # temp[i] - s

# divide by L[i, i]
mul $t2, $t0, $s0
add $t2, $t2, $t0
sll $t2, $t2, 2

la  $t3, L_matrix
add $t2, $t2, $t3
l.s $f14, 0($t2) # Value of L[i, i]

div.s $f12, $f12, $f14 # (temp[i] - s) / L[i, i]

sll $t2, $t0, 2
la  $t3, optimize_coefficient
add $t2, $t3, $t2 # address of x[i]

s.s $f12, 0($t2) # Load (temp[i] - s) / L[i, i] --------> x[i]

addi $t0, $t0, -1
j    back_i

chol_return:
	lw   $ra, 4($sp)
	lw   $s0, 0($sp)
	addi $sp, $sp, 8
	jr   $ra

# =====================================================================
# FUNCTION : filter_signal
# PURPOSE  : Apply FIR filter h[k] (hopt) to input x[n] (input) with zero-padding.
# FORMULA  : y[n] = sum_{k=0..M-1} h[k] * x[n - k], with x[idx<0]=0
# ARGUMENTS:
# $a0 = N (length of x)
# $a1 = M (length of h)
# RETURNS:
# Writes y[n] (float) to y_array
# =====================================================================
filter_signal:
	addi $sp, $sp, -16         # allocate stack frame
	sw   $ra, 12($sp)
	sw   $s0, 8($sp)
	sw   $s1, 4($sp)
	sw   $s2, 0($sp)

	la   $t0, input_signal     # base address of x (input)
	la   $t1, optimize_coefficient        # base address of h (filter)
	la   $t2, output_signal     # base address of y (output)
	move $s0, $zero            # n = 0

loop_n:
	bge $s0, $a0, filter_done  # if n >= N: done

# s = 0.0
mtc1 $zero, $f0

move $s1, $zero            # k = 0

loop_k:
	bge $s1, $a1, store_y      # if k >= M: go store y[n]

# idx = n - k
sub $s2, $s0, $s1

# skipping
bltz $s2, skip_k           # skip if idx < 0
bge  $s2, $a0, skip_k       # skip if idx >= N

# Load h[k]
mul $t3, $s1, 4
add $t3, $t1, $t3
l.s $f2, 0($t3)

# Load x[idx]
mul $t3, $s2, 4
add $t3, $t0, $t3
l.s $f4, 0($t3)

# s += h[k] * x[idx]
mul.s $f6, $f2, $f4
add.s $f0, $f0, $f6

skip_k:
	addi $s1, $s1, 1
	j    loop_k

store_y:
# y[n] = s
mul $t3, $s0, 4
add $t3, $t2, $t3
s.s $f0, 0($t3)

addi $s0, $s0, 1
j    loop_n

filter_done:
	lw   $ra, 12($sp)
	lw   $s0, 8($sp)
	lw   $s1, 4($sp)
	lw   $s2, 0($sp)
	addi $sp, $sp, 16
	jr   $ra

# -------------------------------------------------------------------------------------
# FUNCTION: compute_mmse
# -------------------------------------------------------------------------------------
# PURPOSE:
# Compute the mean squared error between the global arrays 'desired' and 'output_signal'.
# MMSE = (1/N) * sum_{n=0..N-1} ( desired[n] - output_signal[n] )^2
#
# INPUT:
# $a0 = N    # number of samples
#
# GLOBALS USED:
# desired_signal   - float array (length >= N)
# output_signal     - float array (length >= N)
# mmse  - float scalar to store the result
#
# OUTPUT:
# Stores the computed MMSE (float) into mmse.
#
# RETURN:
# None (returns to caller)
# -------------------------------------------------------------------------------------
compute_mmse:
	addi $sp, $sp, -16       # create stack frame
	sw   $ra, 12($sp)
	sw   $s0, 8($sp)
	sw   $s1, 4($sp)
	sw   $s2, 0($sp)

	move $s0, $a0            # s0 = N
	la   $s1, desired_signal        # s1 = base address of desired[]
	la   $s2, output_signal          # s2 = base address of output_signal[]

# accumulator s = 0.0
mtc1 $zero, $f0
li   $t0, 0              # t0 = n (index)

mmse_loop:
	bge $t0, $s0, mmse_done   # if n >= N -> done

# byte offset = n * 4
sll $t1, $t0, 2

# load desired[n] into $f2
add  $t2, $s1, $t1
lwc1 $f2, 0($t2)

# load output_signal[n] into $f4
add  $t3, $s2, $t1
lwc1 $f4, 0($t3)

# diff = desired[n] - output_signal[n]
sub.s $f6, $f2, $f4

# sq = diff * diff
mul.s $f8, $f6, $f6

# s += sq
add.s $f0, $f0, $f8

addi $t0, $t0, 1
j    mmse_loop

mmse_done:
# convert N (int) to float and divide
mtc1    $s0, $f10
cvt.s.w $f10, $f10
div.s   $f12, $f0, $f10      # f12 = s / N

# store result into mmse
la   $t4, mmse
swc1 $f12, 0($t4)

# restore and return
lw   $ra, 12($sp)
lw   $s0, 8($sp)
lw   $s1, 4($sp)
lw   $s2, 0($sp)
addi $sp, $sp, 16
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

# call round to 1 digit ($f12 as arg -> result: $f0)
jal   round1dp
mov.s $f12, $f0

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
	li $v0, 4
	la $a0, newline
	syscall

	lw   $ra, 4($sp)
	lw   $s0, 0($sp)
	addi $sp, $sp, 8
	jr   $ra

# =====================================================================
# HELPER-UTIL : round1dp
# PURPOSE  : Round a floating-point number to 1 decimal place.
# METHOD   : Multiply by 10, round to nearest integer, then divide by 10.
# FORMULA  : result = round(x * 10) / 10
# ARGUMENTS:
# $f12 = input float value (x)
# RETURNS:
# $f0  = rounded float value (to 1 decimal place)
# =====================================================================
round1dp:
	l.s $f20, ten_float

	mul.s     $f14, $f12, $f20
	round.w.s $f16, $f14
	cvt.s.w   $f18, $f16
	div.s     $f0, $f18, $f20     # <-- use $f0, not $f12

	jr $ra

# =====================================================================
# FUNCTION : read_input_and_desired_from_files
# PURPOSE  : Read numeric text values from "input.txt" + "desired.txt" and load them
# into the global array input_signal.
# =====================================================================
read_input_and_desired_from_files:
    addi $sp, $sp, -28
    sw   $ra, 24($sp)
    sw   $s0, 20($sp)   # N
    sw   $s1, 16($sp)   # fd
    sw   $s2, 12($sp)   # bytes read
    sw   $s3,  8($sp)   # input count
    sw   $s4,  4($sp)   # desired count
    sw   $s5,  0($sp)   # flags: bit0 input, bit1 desired
    lw   $s0, N_word
    move $s3, $zero
    move $s4, $zero
    move $s5, $zero     # flags = 0
    # read input.txt
    li   $v0, 13
    la   $a0, input_file
    li   $a1, 0         # flags = read only
    li   $a2, 0         # mode = default
    syscall
    move $s1, $v0   # fd
    bltz $s1, read_desired
    # syscall read
    li   $v0, 14
    move $a0, $s1       # a0 = fd
    la   $a1, read_buf  # a1 = buffer
    li   $a2, 1024      # a2 = max bytes
    syscall
    move $s2, $v0
    # syscall close
    li   $v0, 16
    move $a0, $s1
    syscall
    blez $s2, read_desired
    # parse buffer to floats
    la   $a0, read_buf
    move $a1, $s2
    la   $a2, input_signal
    move $a3, $s0
    jal  parse_buffer_to_floats
    move $s3, $v0        # count
    move $t0, $v1        # overflow?
    beqz $t0, read_desired
    ori  $s5, $s5, 0x1   # set bit0 if overflow

read_desired:
    # read desired.txt
    li   $v0, 13
    la   $a0, desired_file
    li   $a1, 0
    li   $a2, 0
    syscall
    move $s1, $v0
    bltz $s1, validate_sizes
    # syscall read
    li   $v0, 14
    move $a0, $s1
    la   $a1, read_buf
    li   $a2, 1024
    syscall
    move $s2, $v0
    # syscall close
    li   $v0, 16
    move $a0, $s1
    syscall
    blez $s2, validate_sizes
    # parse buffer to floats
    la   $a0, read_buf
    move $a1, $s2
    la   $a2, desired_signal
    move $a3, $s0
    jal  parse_buffer_to_floats
    move $s4, $v0
    move $t0, $v1
    beqz $t0, validate_sizes
    ori  $s5, $s5, 0x2   # set bit1 if overflow

validate_sizes:
    lw   $t0, N_word
    bne  $s3, $t0, size_error
    bne  $s4, $t0, size_error
    bnez $s5, size_error
    j    read_done

print_then_exit:
    # print to console
    li   $v0, 4
    la   $a0, input_size_error
    syscall
    li   $v0, 10
    syscall

read_done:
    lw   $ra, 24($sp)
    lw   $s0, 20($sp)
    lw   $s1, 16($sp)
    lw   $s2, 12($sp)
    lw   $s3,  8($sp)
    lw   $s4,  4($sp)
    lw   $s5,  0($sp)
    addi $sp, $sp, 28
    jr   $ra

# =====================================================================
# HELPER: parse_buffer_to_floats
# PURPOSE:
#   Parse up to 'max_count' floating-point numbers from a text buffer and
#   store them into a destination float array. Accepts separators:
#   space, tab, newline, carriage return, and comma.
# INPUT:
#   a0 = buf base
#   a1 = bytes count
#   a2 = dest base (float*)
#   a3 = max_count (N)
# OUTPUT:
#   Stores up to N floats into dest
# =====================================================================
parse_buffer_to_floats:
    addi $sp, $sp, -40
    sw   $ra, 36($sp)
    sw   $s0, 32($sp)   # current pointer
    sw   $s1, 28($sp)   # end pointer
    sw   $s2, 24($sp)   # destination pointer
    sw   $s3, 20($sp)   # parsed count
    sw   $s4, 16($sp)   # int part
    sw   $s5, 12($sp)   # fraction part
    sw   $s6,  8($sp)   # fraction length
    sw   $s7,  4($sp)   # sign scalar
    sw   $t9,  0($sp)   # overflow flag local
    # initialize working pointers and counters
    move $s0, $a0       # current pointer = buffer base
    addu $s1, $a0, $a1  # end pointer = buffer base + length
    move $s2, $a2       # destination pointer = destination base
    move $s3, $zero     # parsed count = 0
    move $t9, $zero     # overflow flag local = 0

pb_loop:
    bge  $s0, $s1, pb_done
    bge  $s3, $a3, pb_overflow_scan
    lbu  $t0, 0($s0)    # read current character
    # skip separator characters
    li $t1, 32       # ' '
    beq $t0, $t1, pb_skip
    li $t1, 9        # '\t'
    beq $t0, $t1, pb_skip
    li $t1, 10       # '\n'
    beq $t0, $t1, pb_skip
    li $t1, 13       # '\r'
    beq $t0, $t1, pb_skip
    li $t1, 44       # ','
    beq $t0, $t1, pb_skip
    # parse sign
    li   $s7, 1
    li   $t1, 45     # '-'
    beq  $t0, $t1, pb_set_negative
    li   $t1, 43     # '+'
    beq  $t0, $t1, pb_set_positive
    j    pb_after_sign

pb_set_negative:
    li   $s7, -1
    addi $s0, $s0, 1
    j    pb_after_sign

pb_set_positive:
    addi $s0, $s0, 1

# # # Parse integer digits # # #
pb_after_sign:
    move $s4, $zero # int part

pb_int:
    bge  $s0, $s1, pb_fraction_check
    lbu  $t0, 0($s0)
    li   $t1, 48         # '0'
    blt  $t0, $t1, pb_fraction_check
    li   $t1, 57         # '9'
    bgt  $t0, $t1, pb_fraction_check
    # int part = int part * 10 + digit value
    addi $t0, $t0, -48   # convert ASCII to numeric digit
    mul  $s4, $s4, 10
    addu $s4, $s4, $t0
    # move to next character
    addi $s0, $s0, 1
    j    pb_int

# # # Fraction part # # #
pb_fraction_check:
    move $s5, $zero     # fraction part = 0
    move $s6, $zero     # fraction length = 0
    # skip fraction parsing if not float
    bge  $s0, $s1, pb_build
    lbu  $t0, 0($s0)
    li   $t1, 46        # '.'
    bne  $t0, $t1, pb_build
    # skip the decimal point '.'
    addi $s0, $s0, 1

pb_fraction_loop:
    # stop fraction if end line
    bge  $s0, $s1, pb_build
    lbu  $t0, 0($s0)
    li   $t1, 48         # '0'
    blt  $t0, $t1, pb_build
    li   $t1, 57         # '9'
    bgt  $t0, $t1, pb_build
    # frac = frac * 10 + digit
    addi $t0, $t0, -48
    mul  $s5, $s5, 10
    addu $s5, $s5, $t0
    addi $s6, $s6, 1
    addi $s0, $s0, 1
    j    pb_fraction_loop

# # # Build float value # # #
pb_build:
    # convert int part to float
    mtc1    $s4, $f0
    cvt.s.w $f0, $f0
    # skip if no fraction
    beqz $s6, pb_apply_sign
    # convert fraction part to float
    mtc1    $s5, $f2
    cvt.s.w $f2, $f2
    li      $t0, 0      
    l.s     $f4, ten_float       # f4 = 10.0

pb_div_fraction:
    beq  $t0, $s6, pb_add_fraction
    div.s $f2, $f2, $f4
    addi $t0, $t0, 1
    j    pb_div_fraction

pb_add_fraction:
    add.s $f0, $f0, $f2

# # # Apply sign and store # # #
pb_apply_sign:
    bltz $s7, pb_negative
    j    pb_store

pb_negative:
    neg.s $f0, $f0

pb_store:
    # store the float value into destination array and increase parsed count
    sll  $t0, $s3, 2    # byte offset = parsed count * 4
    add  $t1, $s2, $t0  # destination address = destination base + byte offset
    swc1 $f0, 0($t1)    # write float
    addi $s3, $s3, 1    # parsed count++
    j    pb_loop

# # # Skip separator # # #
pb_skip:
    addi $s0, $s0, 1             
    j    pb_loop

pb_overflow_scan:
    bge  $s0, $s1, pb_done              # stop if end of buffer
    lbu  $t0, 0($s0)
    li   $t1, 48                        # '0'
    blt  $t0, $t1, pb_overflow_next
    li   $t1, 57                        # '9'
    bgt  $t0, $t1, pb_overflow_next
    li   $t9, 1                         # overflow flag local = 1 (extra number exists)
    j    pb_done

pb_overflow_next:
    addi $s0, $s0, 1
    j    pb_overflow_scan

pb_done:
    move $v0, $s3        # return parsed count
    move $v1, $t9        # return overflow flag local
    lw   $ra, 36($sp)
    lw   $s0, 32($sp)
    lw   $s1, 28($sp)
    lw   $s2, 24($sp)
    lw   $s3, 20($sp)
    lw   $s4, 16($sp)
    lw   $s5, 12($sp)
    lw   $s6,  8($sp)
    lw   $s7,  4($sp)
    lw   $t9,  0($sp)
    addi $sp, $sp, 40
    jr   $ra
    addi $s0, $s0, 1
    j    pb_loop

# # # Scan rest to check if any numbers exist after reaching N # # #
scan_extra_numbers:
    bge  $s0, $s1, parse_done
    lbu  $t0, 0($s0)
    li   $t1, 48
    blt  $t0, $t1, scan_skip_character
    li   $t1, 57
    bgt  $t0, $t1, scan_skip_character
    li   $t9, 1 
    j    parse_done

scan_skip_character:
    addi $s0, $s0, 1
    j    scan_extra_numbers

parse_done:
    move $v0, $s3        # count
    move $v1, $t9        # overflow flag
    lw   $ra, 36($sp)
    lw   $s0, 32($sp)
    lw   $s1, 28($sp)
    lw   $s2, 24($sp)
    lw   $s3, 20($sp)
    lw   $s4, 16($sp)
    lw   $s5, 12($sp)
    lw   $s6,  8($sp)
    lw   $s7,  4($sp)
    lw   $t9,  0($sp)
    addi $sp, $sp, 40
    jr   $ra

# =====================================================================
# FUNCTION : write_output_signal_to_file
# PURPOSE  : Write the global array "output_signal" (length = N_word)
# to a text file "output.txt" with values rounded to 1dp.
# =====================================================================
write_output_signal_to_file:
    addi $sp, $sp, -24
    sw   $ra, 20($sp)
    sw   $s0, 16($sp)   # N
    sw   $s1, 12($sp)   # fd
    sw   $s2,  8($sp)   # base address of y
    sw   $s3,  4($sp)   # loop index
    sw   $s4,  0($sp)   # temp
    # open output.txt
    li   $v0, 13
    la   $a0, output_file
    li   $a1, 1
    syscall
    move $s1, $v0
    bltz $s1, open_error_return
    # load variables
    lw   $s0, N_word
    la   $s2, output_signal
    move $s3, $zero		# n = 0
    # print "Filtered output: "
    li   $v0, 15
    move $a0, $s1	# fd
    la   $a1, lbl_filteredoutput
    li   $a2, 18
    syscall

# # # Filtered output # # #
write_output_loop:
    bge  $s3, $s0, write_mmse
    # round y[n] to 1 decimal point
    sll  $t0, $s3, 2    # t0 = byte offset
    add  $t1, $s2, $t0  # t1 = address of y[n]
    lwc1 $f12, 0($t1)
    jal  round1dp	    # $f0 = rounded result
    # multiply by 10 to shift decimal place
    l.s  $f14, ten_float
    mul.s $f16, $f0, $f14
    round.w.s $f18, $f16 	
    mfc1 $t2, $f18	# t2 = int result
    # build string float "[-]ddd.d"
    la   $t5, num_buf   # t5 = buffer pointer
    move $t6, $zero		# t6 = len = 0
    # check if number is negative
    bltz $t2, write_negative_result
    j write_positive_result

write_negative_result:
    neg  $t3, $t2
    li   $t4, 45    # ASCII '-'
    sb   $t4, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    j    handle_integer_and_fraction

write_positive_result:
    move $t3, $t2
    j    handle_integer_and_fraction

handle_integer_and_fraction:
    li   $t4, 10
    div  $t3, $t4
    mflo $t7    # t7 = int part = t3 / 10
    mfhi $t8    # t8 = fraction digit = t3 % 10
    # print int part
    bnez $t7, handle_int_part
    li   $t9, 48    # ASCII for '0'
    sb   $t9, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    j    handle_fraction_part

handle_fraction_part:
    # store the decimal point '.'
    li   $t2, 46    # ASCII for '.'
    sb   $t2, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    # store the fractional digit
    addi $t2, $t8, 48   # convert fraction to ASCII
    sb   $t2, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    # store space after the number
    li   $t2, 32    # ASCII for ' '
    sb   $t2, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    # write buffer to file
    li   $v0, 15
    move $a0, $s1   # fd
    la   $a1, num_buf
    move $a2, $t6
    syscall
    # increment loop index n++
    addi $s3, $s3, 1
    j    write_output_loop

handle_int_part:
    li   $t9, 1
    j power_found

power_found:
    move $t0, $t7
    div  $t0, $t9
    mflo $t0
    li   $t1, 10
    blt  $t0, $t1, print_int_digits
    mul  $t9, $t9, $t1
    j    power_found

print_int_digits:
    beq  $t9, $zero, handle_fraction_part
    move $t0, $t7
    div  $t0, $t9
    mflo $t2
    mfhi $t7
    addi $t2, $t2, 48   # convert int to ASCII
    sb   $t2, 0($t5)    # store digit in num_buf
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    li   $t1, 10
    div  $t9, $t1
    mflo $t9
    j    print_int_digits

# # # MMSE # # #
write_mmse:
    # write prefix "MMSE: "
    li   $v0, 15
    move $a0, $s1
    la   $a1, lbl_mmse
    li   $a2, 7
    syscall
    # write MMSE
    la   $t0, mmse
    lwc1 $f12, 0($t0)
    jal  round1dp
    l.s  $f14, ten_float
    mul.s $f16, $f0, $f14
    round.w.s $f18, $f16
    mfc1 $t2, $f18  # t2 = mmse * 10
    # build string "[-]ddd.d\n"
    la   $t5, num_buf
    move $t6, $zero
    bltz $t2, negative_mmse
    j    positive_mmse

negative_mmse:
    neg  $t3, $t2
    li   $t4, 45    # ASCII for '-'
    sb   $t4, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    # write MMSE digits
    li   $t4, 10
    div  $t3, $t4
    mflo $t7
    mfhi $t8
    bnez $t7, mmse_int_part
    li   $t9, 48    # ASCII for '0'
    sb   $t9, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    j    mmse_fraction_part

positive_mmse:
    move $t3, $t2
    li   $t4, 10
    div  $t3, $t4
    mflo $t7    # integer part
    mfhi $t8    # fraction digit
    bnez $t7, mmse_int_part
    li   $t9, 48    # ASCII for '0'
    sb   $t9, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    j    mmse_fraction_part

mmse_int_part:
    li  $t9, 1
    j   mmse_find_pow

mmse_find_pow:
    move $t0, $t7
    div  $t0, $t9
    mflo $t0
    li   $t1, 10
    blt  $t0, $t1, mmse_print_digits
    mul  $t9, $t9, $t1
    j    mmse_find_pow

mmse_print_digits:
    beq  $t9, $zero, mmse_fraction_part
    move $t0, $t7
    div  $t0, $t9
    mflo $t2
    mfhi $t7
    addi $t2, $t2, 48
    sb   $t2, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    li   $t1, 10
    div  $t9, $t1
    mflo $t9
    j    mmse_print_digits

mmse_fraction_part:
    li   $t2, 46    # ASCII for '.'
    sb   $t2, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    addi $t2, $t8, 48   # fraction digit
    sb   $t2, 0($t5)
    addi $t5, $t5, 1
    addi $t6, $t6, 1
    li   $t2, 10
    sb   $t2, 0($t5)
    addi $t6, $t6, 1
    # write MMSE to file
    li   $v0, 15
    move $a0, $s1
    la   $a1, num_buf
    move $a2, $t6
    syscall
    # close file
    li   $v0, 16
    move $a0, $s1
    syscall
    # restore registers and return
    lw   $ra, 20($sp)
    lw   $s0, 16($sp)
    lw   $s1, 12($sp)
    lw   $s2,  8($sp)
    lw   $s3,  4($sp)
    lw   $s4,  0($sp)
    addi $sp, $sp, 24
    jr   $ra

# # # Error handler # # #
size_error:
    # write "Error: size not match\n" to output.txt
    li   $v0, 13
    la   $a0, output_file
    li   $a1, 1       
    syscall
    move $t0, $v0   # fd
    bltz $t0, print_then_exit
    # syscall write
    li   $v0, 15
    move $a0, $t0   # fd
    la   $a1, input_size_error
    li   $a2, 22    # len("Error: size not match\n")
    syscall
    # syscall close
    li   $v0, 16
    move $a0, $t0
    syscall
    j    print_then_exit

open_error_return:
    # if open failed, return quietly
    lw   $ra, 20($sp)
    lw   $s0, 16($sp)
    lw   $s1, 12($sp)
    lw   $s2,  8($sp)
    lw   $s3,  4($sp)
    lw   $s4,  0($sp)
    addi $sp, $sp, 24
    jr   $ra