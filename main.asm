# ==================================================================================================
# PROJECT  :   ̶S̶E̶K̶A̶I̶ Wiener Filter (M=5, N=10)
# FILE     :  main.asm
# AUTHOR   :  <Your Name or Team Name>
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
# [DEMO SIGNALS]
# --------------------------------------------------------------------------------------------------
input_sig: .float 99.5, -1.2, 10.4, 1.2, 2434.5, 11.5, -0.5, 67.0, -1.5, 1.2

desired:
	.float 100.0, -1.2, 10.9, 1.1, 2435.3, 11.0, -0.8, 66.0, -0.9, 1.1

# --------------------------------------------------------------------------------------------------
# [WORKSPACE ARRAYS]
# --------------------------------------------------------------------------------------------------
gamma_xx:  .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0    # Autocorrelation results (length M)
gamma_dx:  .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # Cross-correlation (future use)

R:
	.float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	.float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

hopt:
	.float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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
la $t2, M_word
lw $s1, 0($t2)          # t3 = M

# --- Call estimate_autocorrelation ---
la   $a0, input_sig
move $a1, $s0
move $a2, $s1
jal  estimate_autocorrelation

# --- Call estimate_crosscorrelation ---
la   $a0, desired     # base of d[n]
la   $a1, input_sig       # base of x[n]
move $a2, $s0             # N
move $a3, $s1             # maxlag
jal  estimate_crosscorrelation

# Build R from gamma_xx
la   $a0, gamma_xx    # gamma_xx base
move $a1, $s1         # M
la   $a2, R           # R base (output)
jal  build_R_from_gamma_xx

# [DEBUG]: print gamma_xx array
la $a0, gamma_xx

move $a1, $s1             # length = M
jal  print_float_array

# [DEBUG]: print gamma_dx
la $a0, gamma_dx

move $a1, $s1
jal  print_float_array

# [DEBUG]: print R as M*M floats for debug:
la $a0, R

mul $a1, $s1, $s1    # a1 = M * M  (pseudo-instr; MARS supports this)
jal print_float_array

# === OWARI ============================================
# Exit
li $v0, 10
syscall

# =====================================================================================================================================
# = [2] CORE FUNCTIONS
# =====================================================================================================================================

# ---------------------------------------------------------
# (core-func) estimate_autocorrelation(x[], N, M)

# Computes:
# gamma_xx[k] = (1/N) * Σ (x[n] * x[n-k])
#
# Inputs:
# a0 = base address of input signal x[]
# a1 = N (int)
# a2 = M (int)

# Output:
# Writes gamma_xx[k] (float) results into memory
# ---------------------------------------------------------
estimate_autocorrelation:
	addi $sp, $sp, -16
	sw   $ra, 12($sp)
	sw   $s0, 8($sp)
	sw   $s1, 4($sp)

	move $s0, $a0    # s0 = base of x
	move $s1, $a1    # s1 = N
	move $s2, $a2    # s2 = M

# for (k = 0; k < M; k++)
li $t0, 0      # k = 0

loop_k:
	bge $t0, $s2, end_est_autocorr

# s = 0.0  (we use $f2 to accumulate)
la  $t2, zero_float
l.s $f2, 0($t2)

# for (n = 0; n < N; n++)
li $t1, 0      # n = 0

loop_n:
	bge $t1, $s1, end_loop_n

# idx = n - k
sub $t2, $t1, $t0
blt $t2, $zero, skip_product
bge $t2, $s1, skip_product

# load x[n] into $f4
sll  $t3, $t1, 2
add  $t4, $s0, $t3
lwc1 $f4, 0($t4)

# load x[idx] into $f6
sll  $t5, $t2, 2
add  $t6, $s0, $t5
lwc1 $f6, 0($t6)

mul.s $f8, $f4, $f6    # f8 = x[n] * x[idx]
add.s $f2, $f2, $f8    # accumulate into f2

skip_product:
	addi $t1, $t1, 1
	j    loop_n

end_loop_n:

# denom = N  (biased estimator)
# gamma_xx[k] = f2 / N

mtc1    $s1, $f14 # move int N to $f14
cvt.s.w $f14, $f14    # convert integer to float: now f14 = float(N)
div.s   $f2, $f2, $f14   # f2 = f2 * (1/N)

# store f2 into gamma_xx[k]
la   $t7, gamma_xx
sll  $t8, $t0, 2
add  $t9, $t7, $t8
swc1 $f2, 0($t9)

addi $t0, $t0, 1
j    loop_k

end_est_autocorr:
	lw   $ra, 12($sp)
	lw   $s0, 8($sp)
	lw   $s1, 4($sp)
	addi $sp, $sp, 16
	jr   $ra

# ---------------------------------------------------------
# estimate_crosscorrelation

# Computes:
# gamma_dx[k] = (1/N) * Σ (d[n] * x[n-k])
#
# Inputs:
# a0 = base address of d[n]   (float array)
# a1 = base address of x[n]   (float array)
# a2 = N                      (int)
# a3 = maxlag                 (int)
#
# Output:
# gamma_dx(l) array stored in gamma_dx
# ---------------------------------------------------------
estimate_crosscorrelation:
	addi $sp, $sp, -16
	sw   $ra, 12($sp)
	sw   $s0, 8($sp)    # l
	sw   $s1, 4($sp)    # n
	sw   $s2, 0($sp)    # temp addr

	la $t4, gamma_dx  # base address of output array
	li $s0, 0         # l = 0

crosscorr_outer_loop:
	bgt $s0, $a3, crosscorr_done

	la  $t2, zero_float
	l.s $f0, 0($t2)

	li $s1, 0         # n = 0

crosscorr_inner_loop:
	bge $s1, $a2, crosscorr_sum_done

	sub $t0, $s1, $s0  # idx = n - l

	bltz $t0, crosscorr_next_n
	bge  $t0, $a2, crosscorr_next_n  # skip if idx >= N

# Load d[n]
sll  $t1, $s1, 2
add  $t2, $a0, $t1
lwc1 $f2, 0($t2)

# Load x[idx]
sll  $t3, $t0, 2
add  $t2, $a1, $t3
lwc1 $f4, 0($t2)

# Multiply and accumulate: s += d[n] * x[idx]
mul.s $f6, $f2, $f4
add.s $f0, $f0, $f6

crosscorr_next_n:
	addi $s1, $s1, 1
	j    crosscorr_inner_loop

crosscorr_sum_done:
# Divide by N (biased version)
mtc1    $a2, $f8
cvt.s.w $f8, $f8
div.s   $f10, $f0, $f8

# Store gamma_dx[l]
sll  $t5, $s0, 2
add  $t6, $t4, $t5
swc1 $f10, 0($t6)

# next l
addi $s0, $s0, 1
j    crosscorr_outer_loop

crosscorr_done:
	lw   $ra, 12($sp)
	lw   $s0, 8($sp)
	lw   $s1, 4($sp)
	lw   $s2, 0($sp)
	addi $sp, $sp, 16
	jr   $ra

# ---------------------------------------------------------
# build_R_from_gamma_xx

# Builds MxM Toeplitz matrix R from gamma_xx array.

# Inputs:
# a0 = base of gamma_xx (float)
# a1 = M (int)
# a2 = base of R (float)   <-- output (row-major)

# Save results to R
# ---------------------------------------------------------
build_R_from_gamma_xx:
	addi $sp, $sp, -16
	sw   $ra, 0($sp)
	sw   $s0, 4($sp)
	sw   $s1, 8($sp)
	sw   $s2, 12($sp)

	move $s0, $a0       # s0 = gamma_xx base
	move $s1, $a1       # s1 = M
	move $s2, $a2       # s2 = R base

	li $t0, 0         # l = 0

build_R_outer:
	bge $t0, $s1, build_R_done    # if l >= M done

	li $t1, 0         # k = 0

build_R_inner:
	bge $t1, $s1, build_R_next_l  # if k >= M -> next l

# compute lag = l - k
sub $t6, $t0, $t1
abs $t6, $t6

# if lag < gamma_len load gamma_xx[lag], else use 0.0
blt $t6, $s1, build_R_load_gamma

# load 0.0
la  $t9, zero_float
l.s $f0, 0($t9)
j   build_R_store

build_R_load_gamma:
	sll  $t7, $t6, 2        # byte offset = lag * 4
	add  $t8, $s0, $t7
	lwc1 $f0, 0($t8)

build_R_store:

# compute index = l * M + k
mul  $t7, $t0, $s1     # t7 = l * M
addu $t7, $t7, $t1    # t7 += k

sll $t7, $t7, 2      # byte offset = index * 4
add $t7, $s2, $t7

swc1 $f0, 0($t7)

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

