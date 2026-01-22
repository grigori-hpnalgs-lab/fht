#!/usr/bin/env python3
"""
Native 128-bit NEON code generator for Fast Walsh-Hadamard Transform - V7.

V7: Unified Parameterized Generator with Extended Grid Search

This version combines the best elements from:
- V3: greedy_merged algorithm for level fusion
- V4: Radix-4 butterfly operations
- V5: KernelParams dataclass pattern
- V6: FFHT-style exhaustive threshold search

NEW in V7:
- Radix control: 2 or 4 (process 1 or 2 levels per pass)
- Unroll factor: 1, 2, 4, 8 (inner loop unrolling)
- Prefetch configuration: distance and hint (L1/L2/none)
- Max registers control: 8 or 16

Output: fht_neon_v7.h (generated via optimize_v7_grid.py)
"""

from dataclasses import dataclass
from typing import Optional, List, Callable
import sys

VERSION = "v7"
max_log_n = 30


@dataclass
class KernelParams:
    """
    Parameters that define a V7 kernel variant.

    Grid search will explore combinations of these to find optimal per-size.
    """
    # Strategy
    strategy: str = 'iterative'  # 'iterative' | 'recursive'
    threshold: int = 14          # Base case size for recursive (2..log_n)

    # Radix control
    radix: int = 2               # 2 or 4 (levels processed per composite step)

    # Loop tuning
    unroll_factor: int = 1       # 1, 2, 4, 8 (inner loop body duplication)

    # Memory prefetch
    prefetch_distance: int = 0   # 0 = disabled, or 256, 512, 1024 bytes
    prefetch_hint: int = 3       # 3=L1, 2=L2, 1=L3, 0=non-temporal

    # Register pressure
    max_registers: int = 16      # 8 or 16 (limit to reduce spills)

    def __post_init__(self):
        """Validate parameters."""
        assert self.strategy in ('iterative', 'recursive'), f"Invalid strategy: {self.strategy}"
        assert self.radix in (2, 4), f"Invalid radix: {self.radix}"
        assert self.unroll_factor in (1, 2, 4, 8), f"Invalid unroll_factor: {self.unroll_factor}"
        assert self.prefetch_distance >= 0, f"Invalid prefetch_distance: {self.prefetch_distance}"
        assert self.prefetch_hint in (0, 1, 2, 3), f"Invalid prefetch_hint: {self.prefetch_hint}"
        assert self.max_registers in (8, 16), f"Invalid max_registers: {self.max_registers}"

    def to_dict(self):
        return {
            'strategy': self.strategy,
            'threshold': self.threshold,
            'radix': self.radix,
            'unroll_factor': self.unroll_factor,
            'prefetch_distance': self.prefetch_distance,
            'prefetch_hint': self.prefetch_hint,
            'max_registers': self.max_registers,
        }

    def short_desc(self) -> str:
        """Short description for logging."""
        parts = [self.strategy]
        if self.strategy == 'recursive':
            parts.append(f"th={self.threshold}")
        if self.radix != 2:
            parts.append(f"r{self.radix}")
        if self.unroll_factor != 1:
            parts.append(f"u{self.unroll_factor}")
        if self.prefetch_distance > 0:
            parts.append(f"pf{self.prefetch_distance}")
        if self.max_registers != 16:
            parts.append(f"reg{self.max_registers}")
        return ",".join(parts)


# =============================================================================
# Float NEON Level-0 Butterfly (in-register, distance 1)
# =============================================================================

def float_neon_0(register: str, aux_registers: List[str], ident: str = '') -> str:
    """
    Level 0 butterfly: pairs at distance 1.
    Input:  [a, b, c, d]
    Output: [a+b, a-b, c+d, c-d]
    """
    res = ident + '{\n'
    res += ident + '  float32x4_t t0 = vrev64q_f32(%s);\n' % register
    res += ident + '  float32x4_t t1 = vaddq_f32(%s, t0);\n' % register
    res += ident + '  t0 = vsubq_f32(%s, t0);\n' % register
    res += ident + '  %s = vtrn1q_f32(t1, t0);\n' % register
    res += ident + '}\n'
    return res


# =============================================================================
# Float NEON Level-1 Butterfly (in-register, distance 2)
# =============================================================================

def float_neon_1(register: str, aux_registers: List[str], ident: str = '') -> str:
    """
    Level 1 butterfly: pairs at distance 2.
    Input:  [a, b, c, d]
    Output: [a+c, b+d, a-c, b-d]
    """
    res = ident + '{\n'
    res += ident + '  float32x4_t t0 = vextq_f32(%s, %s, 2);\n' % (register, register)
    res += ident + '  float32x4_t t1 = vaddq_f32(%s, t0);\n' % register
    res += ident + '  t0 = vsubq_f32(%s, t0);\n' % register
    res += ident + '  %s = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t0)));\n' % register
    res += ident + '}\n'
    return res


# =============================================================================
# Float NEON Level-2+ Butterfly (inter-register, distance 4+)
# =============================================================================

def float_neon_2_etc(from_r0: str, from_r1: str, to_r0: str, to_r1: str, ident: str = '') -> str:
    """Level 2+: inter-register butterflies."""
    res = ident + '%s = vaddq_f32(%s, %s);\n' % (to_r0, from_r0, from_r1)
    res += ident + '%s = vsubq_f32(%s, %s);\n' % (to_r1, from_r0, from_r1)
    return res


# =============================================================================
# Float Radix-4 In-Register (2 levels at once)
# =============================================================================

def float_radix4_inreg(register: str, ident: str = '') -> str:
    """
    In-register radix-4 for 4 contiguous floats.
    Processes levels 0 and 1 together.

    Input:  [x0, x1, x2, x3]
    Output: [x0+x1+x2+x3, x0-x1+x2-x3, x0+x1-x2-x3, x0-x1-x2+x3]

    Algorithm:
      Level 1 (distance 2): compute sum/diff at distance 2
      Level 0 (distance 1): compute sum/diff at distance 1
    """
    res = ident + '{\n'
    # Step 1: Create rotated version [x2, x3, x0, x1]
    res += ident + '  float32x4_t rot2 = vextq_f32(%s, %s, 2);\n' % (register, register)

    # Step 2: Level 1 - sumdiff at distance 2
    res += ident + '  float32x4_t sum02 = vaddq_f32(%s, rot2);\n' % register
    res += ident + '  float32x4_t dif02 = vsubq_f32(%s, rot2);\n' % register

    # Combine low halves: [s0, s1, d0, d1]
    res += ident + '  float32x4_t combined = vreinterpretq_f32_f64(vzip1q_f64('
    res += 'vreinterpretq_f64_f32(sum02), vreinterpretq_f64_f32(dif02)));\n'

    # Step 3: Level 0 - sumdiff at distance 1
    res += ident + '  float32x4_t rev1 = vrev64q_f32(combined);\n'
    res += ident + '  float32x4_t sum_final = vaddq_f32(combined, rev1);\n'
    res += ident + '  float32x4_t dif_final = vsubq_f32(combined, rev1);\n'

    # Interleave results
    res += ident + '  %s = vtrn1q_f32(sum_final, dif_final);\n' % register
    res += ident + '}\n'
    return res


# =============================================================================
# Float Sumdiff for Two Registers
# =============================================================================

def float_sumdiff_regs(r0: str, r1: str, ident: str = '') -> str:
    """
    In-place sumdiff: r0 = r0 + r1, r1 = r0 - r1
    """
    res = ident + '{\n'
    res += ident + '  float32x4_t t = %s;\n' % r0
    res += ident + '  %s = vaddq_f32(%s, %s);\n' % (r0, r0, r1)
    res += ident + '  %s = vsubq_f32(t, %s);\n' % (r1, r1)
    res += ident + '}\n'
    return res


# =============================================================================
# Float Radix-8 In-Register (3 levels at once)
# =============================================================================

def float_radix8_inreg(r0: str, r1: str, ident: str = '') -> str:
    """
    In-register radix-8 for 8 contiguous floats in 2 registers.
    Processes levels 0, 1, and 2 together.
    """
    res = ident + '// Radix-8: Level 2 (distance 4)\n'
    res += float_sumdiff_regs(r0, r1, ident)
    res += ident + '// Radix-8: Levels 1,0 via radix-4 on each register\n'
    res += float_radix4_inreg(r0, ident)
    res += float_radix4_inreg(r1, ident)
    return res


# =============================================================================
# Double NEON Level-0 Butterfly (in-register, distance 1)
# =============================================================================

def double_neon_0(register: str, aux_registers: List[str], ident: str = '') -> str:
    """
    Level 0 butterfly for float64x2_t: pairs at distance 1.
    Input:  [a, b]
    Output: [a+b, a-b]

    Uses vextq_f64 to rotate elements:
    - rot = [b, a] (rotate by 1)
    - sum = [a+b, b+a]
    - dif = [a-b, b-a]
    - result = interleave low lanes = [a+b, a-b]
    """
    res = ident + '{\n'
    # Rotate to get [b, a]
    res += ident + '  float64x2_t rot = vextq_f64(%s, %s, 1);\n' % (register, register)
    # sum = [a+b, b+a] (both lanes have same value)
    res += ident + '  float64x2_t sum = vaddq_f64(%s, rot);\n' % register
    # dif = [a-b, b-a]
    res += ident + '  float64x2_t dif = vsubq_f64(%s, rot);\n' % register
    # Interleave: take low lane of sum and low lane of dif
    res += ident + '  %s = vtrn1q_f64(sum, dif);\n' % register
    res += ident + '}\n'
    return res


# =============================================================================
# Double NEON Level-1+ Butterfly (inter-register, distance 2+)
# =============================================================================

def double_neon_1_etc(from_r0: str, from_r1: str, to_r0: str, to_r1: str, ident: str = '') -> str:
    """Level 1+: inter-register butterflies for doubles."""
    res = ident + '%s = vaddq_f64(%s, %s);\n' % (to_r0, from_r0, from_r1)
    res += ident + '%s = vsubq_f64(%s, %s);\n' % (to_r1, from_r0, from_r1)
    return res


# =============================================================================
# Double Sumdiff for Two Registers
# =============================================================================

def double_sumdiff_regs(r0: str, r1: str, ident: str = '') -> str:
    """
    In-place sumdiff for doubles: r0 = r0 + r1, r1 = r0 - r1
    """
    res = ident + '{\n'
    res += ident + '  float64x2_t t = %s;\n' % r0
    res += ident + '  %s = vaddq_f64(%s, %s);\n' % (r0, r0, r1)
    res += ident + '  %s = vsubq_f64(t, %s);\n' % (r1, r1)
    res += ident + '}\n'
    return res


# =============================================================================
# Prefetch Code Generation
# =============================================================================

def generate_prefetch(buf_name: str, offset_expr: str, params: KernelParams, ident: str = '') -> str:
    """Generate prefetch instruction if enabled."""
    if params.prefetch_distance <= 0:
        return ''

    return ident + '__builtin_prefetch(%s + %s + %d, 0, %d);\n' % (
        buf_name, offset_expr, params.prefetch_distance, params.prefetch_hint)


# =============================================================================
# Plain Scalar Step (fallback for very small sizes)
# =============================================================================

def plain_step(type_name: str, buf_name: str, log_n: int, it: int, ident: str = '') -> str:
    """Scalar butterfly for one level."""
    n = 1 << log_n
    res = ident + "for (int j = 0; j < %d; j += %d) {\n" % (n, 1 << (it + 1))
    res += ident + "  for (int k = 0; k < %d; ++k) {\n" % (1 << it)
    res += ident + "    %s u = %s[j + k];\n" % (type_name, buf_name)
    res += ident + "    %s v = %s[j + k + %d];\n" % (type_name, buf_name, 1 << it)
    res += ident + "    %s[j + k] = u + v;\n" % buf_name
    res += ident + "    %s[j + k + %d] = u - v;\n" % (buf_name, 1 << it)
    res += ident + "  }\n"
    res += ident + "}\n"
    return res


# =============================================================================
# Composite Step Generator (V7 - parameterized)
# =============================================================================

def composite_step_v7(buf_name: str, log_n: int, from_it: int, to_it: int,
                      params: KernelParams, ident: str = '') -> str:
    """
    Generate code for one composite pass merging levels [from_it, to_it).

    This is the core function that handles:
    - In-register butterflies for levels < 2 (log_w=2 for float)
    - Inter-register butterflies for levels >= 2
    - Optional unrolling
    - Optional prefetching
    """
    log_w = 2  # 4 floats per float32x4_t register
    num_registers = params.max_registers

    if log_n < log_w:
        raise Exception('need at least %d elements' % (1 << log_w))

    if num_registers % 2 == 1:
        raise Exception('odd number of registers')

    # Calculate how many non-trivial (inter-register) levels
    num_nontrivial_levels = 0
    if to_it > log_w:
        first_nontrivial = max(from_it, log_w)
        num_nontrivial_levels = to_it - first_nontrivial
        if 1 << num_nontrivial_levels > num_registers // 2:
            raise Exception('not enough registers')

    n = 1 << log_n
    elements_per_reg = 1 << log_w  # 4 for float32x4_t

    input_regs = ['r%d' % i for i in range(num_registers // 2)]
    output_regs = ['r%d' % i for i in range(num_registers // 2, num_registers)]

    # Simple case: only in-register butterflies (levels 0-1)
    if num_nontrivial_levels == 0:
        # Determine unroll factor (limited by available iterations)
        unroll = min(params.unroll_factor, n // elements_per_reg)
        step = elements_per_reg * unroll

        res = ident + 'for (unsigned long j = 0; j < %d; j += %d) {\n' % (n, step)

        # Prefetch
        if params.prefetch_distance > 0:
            res += generate_prefetch(buf_name, 'j', params, ident + '  ')

        # Load all registers for unrolled iterations
        for u in range(unroll):
            off = u * elements_per_reg
            reg = 'ru%d' % u
            res += ident + '  float32x4_t %s = vld1q_f32(%s + j + %d);\n' % (reg, buf_name, off)

        # Process each register based on radix
        for u in range(unroll):
            reg = 'ru%d' % u
            if params.radix == 4 and from_it == 0 and to_it >= 2:
                # Use radix-4 for levels 0-1 combined
                res += float_radix4_inreg(reg, ident + '  ')
            else:
                # Sequential radix-2 for each level
                for it in range(from_it, to_it):
                    if it == 0:
                        res += float_neon_0(reg, output_regs, ident + '  ')
                    elif it == 1:
                        res += float_neon_1(reg, output_regs, ident + '  ')

        # Store all registers
        for u in range(unroll):
            off = u * elements_per_reg
            reg = 'ru%d' % u
            res += ident + '  vst1q_f32(%s + j + %d, %s);\n' % (buf_name, off, reg)

        res += ident + '}\n'
        return res

    # Complex case: inter-register butterflies needed
    num_active = 1 << num_nontrivial_levels
    outer_stride = 1 << to_it
    inner_stride = 1 << (to_it - num_nontrivial_levels)

    res = ident + 'for (unsigned long j = 0; j < %d; j += %d) {\n' % (n, outer_stride)
    res += ident + '  for (unsigned long k = 0; k < %d; k += %d) {\n' % (inner_stride, elements_per_reg)

    # Declare registers
    for i in range(num_active):
        res += ident + '    float32x4_t %s;\n' % input_regs[i]
    for i in range(num_active):
        res += ident + '    float32x4_t %s;\n' % output_regs[i]

    # Prefetch before loads
    if params.prefetch_distance > 0:
        for l in range(num_active):
            offset = l * inner_stride
            res += ident + '    __builtin_prefetch(%s + j + k + %d + %d, 0, %d);\n' % (
                buf_name, offset, params.prefetch_distance, params.prefetch_hint)

    # Load
    for l in range(num_active):
        offset = l * inner_stride
        res += ident + '    %s = vld1q_f32(%s + j + k + %d);\n' % (
            input_regs[l], buf_name, offset)

    # In-register butterflies (levels < log_w)
    inreg_from = from_it
    inreg_to = min(to_it, log_w)

    if params.radix == 4 and inreg_from == 0 and inreg_to >= 2:
        # Use radix-4 for levels 0-1 combined
        for ii in range(num_active):
            res += float_radix4_inreg(input_regs[ii], ident + '    ')
    else:
        # Sequential radix-2
        for it in range(inreg_from, inreg_to):
            for ii in range(num_active):
                if it == 0:
                    res += float_neon_0(input_regs[ii], output_regs, ident + '    ')
                elif it == 1:
                    res += float_neon_1(input_regs[ii], output_regs, ident + '    ')

    # Inter-register butterflies (levels >= log_w)
    for it in range(num_nontrivial_levels):
        for ii in range(0, num_active, 1 << (it + 1)):
            for jj in range(1 << it):
                res += float_neon_2_etc(input_regs[ii + jj],
                                        input_regs[ii + jj + (1 << it)],
                                        output_regs[ii + jj],
                                        output_regs[ii + jj + (1 << it)],
                                        ident + '    ')
        # Swap input/output registers
        tmp = input_regs
        input_regs = output_regs
        output_regs = tmp

    # Store
    for l in range(num_active):
        offset = l * inner_stride
        res += ident + '    vst1q_f32(%s + j + k + %d, %s);\n' % (
            buf_name, offset, input_regs[l])

    res += ident + '  }\n'
    res += ident + '}\n'
    return res


# =============================================================================
# Composite Step Generator for Double Precision (log_w=1)
# =============================================================================

def composite_step_v7_double(buf_name: str, log_n: int, from_it: int, to_it: int,
                              params: KernelParams, ident: str = '') -> str:
    """
    Generate code for one composite pass for double precision.

    Key differences from float version:
    - log_w = 1 (2 doubles per float64x2_t register)
    - Only level 0 is in-register; level 1+ are inter-register
    - No radix-4 (can't do 2 levels with only 2 elements)
    """
    log_w = 1  # 2 doubles per float64x2_t register
    num_registers = params.max_registers

    if log_n < log_w:
        raise Exception('need at least %d elements' % (1 << log_w))

    if num_registers % 2 == 1:
        raise Exception('odd number of registers')

    # Calculate how many non-trivial (inter-register) levels
    num_nontrivial_levels = 0
    if to_it > log_w:
        first_nontrivial = max(from_it, log_w)
        num_nontrivial_levels = to_it - first_nontrivial
        if 1 << num_nontrivial_levels > num_registers // 2:
            raise Exception('not enough registers')

    n = 1 << log_n
    elements_per_reg = 1 << log_w  # 2 for float64x2_t

    input_regs = ['r%d' % i for i in range(num_registers // 2)]
    output_regs = ['r%d' % i for i in range(num_registers // 2, num_registers)]

    # Simple case: only in-register butterflies (level 0 only)
    if num_nontrivial_levels == 0:
        # Determine unroll factor (limited by available iterations)
        unroll = min(params.unroll_factor, n // elements_per_reg)
        step = elements_per_reg * unroll

        res = ident + 'for (unsigned long j = 0; j < %d; j += %d) {\n' % (n, step)

        # Prefetch
        if params.prefetch_distance > 0:
            res += generate_prefetch(buf_name, 'j', params, ident + '  ')

        # Load all registers for unrolled iterations
        for u in range(unroll):
            off = u * elements_per_reg
            reg = 'ru%d' % u
            res += ident + '  float64x2_t %s = vld1q_f64(%s + j + %d);\n' % (reg, buf_name, off)

        # Process each register - only level 0 can be in-register for doubles
        for u in range(unroll):
            reg = 'ru%d' % u
            for it in range(from_it, to_it):
                if it == 0:
                    res += double_neon_0(reg, output_regs, ident + '  ')
                # Note: level 1 cannot be done in-register for doubles

        # Store all registers
        for u in range(unroll):
            off = u * elements_per_reg
            reg = 'ru%d' % u
            res += ident + '  vst1q_f64(%s + j + %d, %s);\n' % (buf_name, off, reg)

        res += ident + '}\n'
        return res

    # Complex case: inter-register butterflies needed
    num_active = 1 << num_nontrivial_levels
    outer_stride = 1 << to_it
    inner_stride = 1 << (to_it - num_nontrivial_levels)

    res = ident + 'for (unsigned long j = 0; j < %d; j += %d) {\n' % (n, outer_stride)
    res += ident + '  for (unsigned long k = 0; k < %d; k += %d) {\n' % (inner_stride, elements_per_reg)

    # Declare registers
    for i in range(num_active):
        res += ident + '    float64x2_t %s;\n' % input_regs[i]
    for i in range(num_active):
        res += ident + '    float64x2_t %s;\n' % output_regs[i]

    # Prefetch before loads
    if params.prefetch_distance > 0:
        for l in range(num_active):
            offset = l * inner_stride
            res += ident + '    __builtin_prefetch(%s + j + k + %d + %d, 0, %d);\n' % (
                buf_name, offset, params.prefetch_distance, params.prefetch_hint)

    # Load
    for l in range(num_active):
        offset = l * inner_stride
        res += ident + '    %s = vld1q_f64(%s + j + k + %d);\n' % (
            input_regs[l], buf_name, offset)

    # In-register butterflies (only level 0 for doubles)
    inreg_from = from_it
    inreg_to = min(to_it, log_w)

    for it in range(inreg_from, inreg_to):
        for ii in range(num_active):
            if it == 0:
                res += double_neon_0(input_regs[ii], output_regs, ident + '    ')

    # Inter-register butterflies (levels >= log_w, i.e., level 1+)
    for it in range(num_nontrivial_levels):
        for ii in range(0, num_active, 1 << (it + 1)):
            for jj in range(1 << it):
                res += double_neon_1_etc(input_regs[ii + jj],
                                         input_regs[ii + jj + (1 << it)],
                                         output_regs[ii + jj],
                                         output_regs[ii + jj + (1 << it)],
                                         ident + '    ')
        # Swap input/output registers
        tmp = input_regs
        input_regs = output_regs
        output_regs = tmp

    # Store
    for l in range(num_active):
        offset = l * inner_stride
        res += ident + '    vst1q_f64(%s + j + k + %d, %s);\n' % (
            buf_name, offset, input_regs[l])

    res += ident + '  }\n'
    res += ident + '}\n'
    return res


# =============================================================================
# Greedy Merged Generator (V7 - with parameters)
# =============================================================================

def greedy_merged_v7(type_name: str, log_n: int, params: KernelParams,
                     func_name: Optional[str] = None) -> str:
    """
    Generate iterative FHT kernel using greedy level merging.
    Tries to merge as many levels as possible per pass.
    """
    if func_name is None:
        func_name = 'fht_neon_%s_%s_%d' % (VERSION, type_name, log_n)

    # Validate
    try:
        composite_step_v7('buf', log_n, 0, 0, params, '')
    except Exception:
        raise Exception('log_n is too small: %d' % log_n)

    signature = 'static inline void %s(%s *buf)' % (func_name, type_name)
    res = '%s {\n' % signature

    cur_it = 0
    while cur_it < log_n:
        cur_to_it = log_n
        while True:
            try:
                composite_step_v7('buf', log_n, cur_it, cur_to_it, params, '')
                break
            except Exception:
                cur_to_it -= 1
        res += composite_step_v7('buf', log_n, cur_it, cur_to_it, params, '  ')
        cur_it = cur_to_it

    res += '}\n'
    return res


# =============================================================================
# Greedy Merged Recursive Generator (V7 - with parameters)
# =============================================================================

def greedy_merged_recursive_v7(type_name: str, log_n: int, params: KernelParams,
                                func_name: Optional[str] = None) -> str:
    """
    Generate recursive FHT kernel with configurable threshold.
    Uses divide-and-conquer with the threshold as the base case size.
    """
    if func_name is None:
        func_name = 'fht_neon_%s_%s_%d' % (VERSION, type_name, log_n)

    threshold = min(params.threshold, log_n)

    if threshold > log_n:
        raise Exception('threshold must be at most log_n')

    # Validate threshold is valid
    try:
        composite_step_v7('buf', threshold, 0, 0, params, '')
    except Exception:
        raise Exception('threshold is too small')

    # Recursive helper function signature
    rec_func = func_name + '_recursive'
    signature = 'static void %s(%s *buf, int depth)' % (rec_func, type_name)
    res = '%s {\n' % signature

    # Base case: when depth reaches threshold, use iterative
    res += '  if (depth == %d) {\n' % threshold
    cur_it = 0
    while cur_it < threshold:
        cur_to_it = threshold
        while True:
            try:
                # Base case uses no prefetch (small, fits in cache)
                base_params = KernelParams(
                    radix=params.radix,
                    unroll_factor=params.unroll_factor,
                    prefetch_distance=0,  # No prefetch in base case
                    max_registers=params.max_registers
                )
                composite_step_v7('buf', threshold, cur_it, cur_to_it, base_params, '')
                break
            except Exception:
                cur_to_it -= 1
        base_params = KernelParams(
            radix=params.radix,
            unroll_factor=params.unroll_factor,
            prefetch_distance=0,
            max_registers=params.max_registers
        )
        res += composite_step_v7('buf', threshold, cur_it, cur_to_it, base_params, '    ')
        cur_it = cur_to_it
    res += '    return;\n'
    res += '  }\n'

    # Recursive cases: for each level above threshold
    cur_it = threshold
    while cur_it < log_n:
        cur_to_it = log_n
        while True:
            try:
                composite_step_v7('buf', cur_to_it, cur_it, cur_to_it, params, '')
                break
            except Exception:
                cur_to_it -= 1

        res += '  if (depth == %d) {\n' % cur_to_it
        num_subblocks = 1 << (cur_to_it - cur_it)
        for i in range(num_subblocks):
            res += '    %s(buf + %d, %d);\n' % (rec_func, i * (1 << cur_it), cur_it)
        # Merge phase may use prefetch for large sizes
        res += composite_step_v7('buf', cur_to_it, cur_it, cur_to_it, params, '    ')
        res += '    return;\n'
        res += '  }\n'
        cur_it = cur_to_it

    res += '}\n'

    # Wrapper function
    wrapper_sig = 'static inline void %s(%s *buf)' % (func_name, type_name)
    res += '%s {\n' % wrapper_sig
    res += '  %s(buf, %d);\n' % (rec_func, log_n)
    res += '}\n'
    return res


# =============================================================================
# Greedy Merged Generator for Double (V7)
# =============================================================================

def greedy_merged_v7_double(log_n: int, params: KernelParams,
                             func_name: Optional[str] = None) -> str:
    """
    Generate iterative FHT kernel for double precision.
    Similar to float version but uses log_w=1 (2 doubles per register).
    """
    if func_name is None:
        func_name = 'fht_neon_%s_double_%d' % (VERSION, log_n)

    # Validate
    try:
        composite_step_v7_double('buf', log_n, 0, 0, params, '')
    except Exception:
        raise Exception('log_n is too small: %d' % log_n)

    signature = 'static inline void %s(double *buf)' % func_name
    res = '%s {\n' % signature

    cur_it = 0
    while cur_it < log_n:
        cur_to_it = log_n
        while True:
            try:
                composite_step_v7_double('buf', log_n, cur_it, cur_to_it, params, '')
                break
            except Exception:
                cur_to_it -= 1
        res += composite_step_v7_double('buf', log_n, cur_it, cur_to_it, params, '  ')
        cur_it = cur_to_it

    res += '}\n'
    return res


# =============================================================================
# Greedy Merged Recursive Generator for Double (V7)
# =============================================================================

def greedy_merged_recursive_v7_double(log_n: int, params: KernelParams,
                                       func_name: Optional[str] = None) -> str:
    """
    Generate recursive FHT kernel for double precision.
    Uses divide-and-conquer with configurable threshold.
    """
    if func_name is None:
        func_name = 'fht_neon_%s_double_%d' % (VERSION, log_n)

    threshold = min(params.threshold, log_n)

    if threshold > log_n:
        raise Exception('threshold must be at most log_n')

    # Validate threshold is valid
    try:
        composite_step_v7_double('buf', threshold, 0, 0, params, '')
    except Exception:
        raise Exception('threshold is too small')

    # Recursive helper function signature
    rec_func = func_name + '_recursive'
    signature = 'static void %s(double *buf, int depth)' % rec_func
    res = '%s {\n' % signature

    # Base case: when depth reaches threshold, use iterative
    res += '  if (depth == %d) {\n' % threshold
    cur_it = 0
    while cur_it < threshold:
        cur_to_it = threshold
        while True:
            try:
                base_params = KernelParams(
                    radix=2,  # Force radix-2 for double (no radix-4 support)
                    unroll_factor=params.unroll_factor,
                    prefetch_distance=0,  # No prefetch in base case
                    max_registers=params.max_registers
                )
                composite_step_v7_double('buf', threshold, cur_it, cur_to_it, base_params, '')
                break
            except Exception:
                cur_to_it -= 1
        base_params = KernelParams(
            radix=2,
            unroll_factor=params.unroll_factor,
            prefetch_distance=0,
            max_registers=params.max_registers
        )
        res += composite_step_v7_double('buf', threshold, cur_it, cur_to_it, base_params, '    ')
        cur_it = cur_to_it
    res += '    return;\n'
    res += '  }\n'

    # Recursive cases: for each level above threshold
    cur_it = threshold
    while cur_it < log_n:
        cur_to_it = log_n
        while True:
            try:
                composite_step_v7_double('buf', cur_to_it, cur_it, cur_to_it, params, '')
                break
            except Exception:
                cur_to_it -= 1

        res += '  if (depth == %d) {\n' % cur_to_it
        num_subblocks = 1 << (cur_to_it - cur_it)
        for i in range(num_subblocks):
            res += '    %s(buf + %d, %d);\n' % (rec_func, i * (1 << cur_it), cur_it)
        res += composite_step_v7_double('buf', cur_to_it, cur_it, cur_to_it, params, '    ')
        res += '    return;\n'
        res += '  }\n'
        cur_it = cur_to_it

    res += '}\n'

    # Wrapper function
    wrapper_sig = 'static inline void %s(double *buf)' % func_name
    res += '%s {\n' % wrapper_sig
    res += '  %s(buf, %d);\n' % (rec_func, log_n)
    res += '}\n'
    return res


# =============================================================================
# Plain Unmerged Generator (fallback for very small sizes)
# =============================================================================

def plain_unmerged_v7(type_name: str, log_n: int, func_name: Optional[str] = None) -> str:
    """Generate plain scalar implementation."""
    if func_name is None:
        func_name = 'fht_neon_%s_%s_%d' % (VERSION, type_name, log_n)

    signature = "static inline void %s(%s *buf)" % (func_name, type_name)
    res = '%s {\n' % signature
    for i in range(log_n):
        res += plain_step(type_name, 'buf', log_n, i, '  ')
    res += "}\n"
    return res


# =============================================================================
# Main Kernel Generator (V7) - Float
# =============================================================================

def generate_kernel_v7(log_n: int, params: KernelParams,
                       func_name: Optional[str] = None) -> str:
    """
    Generate FHT kernel based on parameters.

    Returns the complete C code for the kernel function.
    """
    if func_name is None:
        func_name = 'fht_neon_%s_float_%d' % (VERSION, log_n)

    # Very small sizes: use scalar
    if log_n <= 1:
        return plain_unmerged_v7('float', log_n, func_name)

    # Choose strategy
    if params.strategy == 'iterative':
        try:
            return greedy_merged_v7('float', log_n, params, func_name)
        except Exception:
            return plain_unmerged_v7('float', log_n, func_name)
    else:  # recursive
        try:
            return greedy_merged_recursive_v7('float', log_n, params, func_name)
        except Exception:
            # Fall back to iterative
            try:
                return greedy_merged_v7('float', log_n, params, func_name)
            except Exception:
                return plain_unmerged_v7('float', log_n, func_name)


# =============================================================================
# Main Kernel Generator (V7) - Double
# =============================================================================

def generate_kernel_double_v7(log_n: int, params: KernelParams,
                               func_name: Optional[str] = None) -> str:
    """
    Generate double precision FHT kernel based on parameters.

    Key differences from float:
    - Uses log_w=1 (2 doubles per register vs 4 floats)
    - No radix-4 support (radix is forced to 2)
    - Level 0 is the only in-register operation

    Returns the complete C code for the kernel function.
    """
    if func_name is None:
        func_name = 'fht_neon_%s_double_%d' % (VERSION, log_n)

    # Very small sizes: use scalar
    if log_n <= 0:
        return plain_unmerged_v7('double', log_n, func_name)

    # Force radix-2 for double precision (can't do radix-4 with 2 elements)
    double_params = KernelParams(
        strategy=params.strategy,
        threshold=params.threshold,
        radix=2,  # Always radix-2 for double
        unroll_factor=params.unroll_factor,
        prefetch_distance=params.prefetch_distance,
        prefetch_hint=params.prefetch_hint,
        max_registers=params.max_registers,
    )

    # Choose strategy
    if double_params.strategy == 'iterative':
        try:
            return greedy_merged_v7_double(log_n, double_params, func_name)
        except Exception:
            return plain_unmerged_v7('double', log_n, func_name)
    else:  # recursive
        try:
            return greedy_merged_recursive_v7_double(log_n, double_params, func_name)
        except Exception:
            # Fall back to iterative
            try:
                return greedy_merged_v7_double(log_n, double_params, func_name)
            except Exception:
                return plain_unmerged_v7('double', log_n, func_name)


# =============================================================================
# Header Generation
# =============================================================================

def generate_header_start() -> str:
    """Generate header file preamble."""
    return f'''#ifndef FHT_NEON_{VERSION.upper()}_H
#define FHT_NEON_{VERSION.upper()}_H

#include <arm_neon.h>

// FHT NEON {VERSION.upper()} - Unified Parameterized Implementation
//
// This version combines:
// - V3's greedy_merged algorithm for level fusion
// - V4's radix-4 butterfly operations
// - V5's KernelParams pattern
// - V6's FFHT-style exhaustive threshold search
//
// NEW optimizations in V7:
// - Radix control (radix-2 vs radix-4 per pass)
// - Inner loop unrolling
// - Configurable prefetch distance and locality hint
// - Register pressure control
// - Double precision support (log_w=1, radix-2 only)
//
// Generated by optimize_v7_grid.py with per-size optimal parameters.

'''


def generate_header_end(max_log_n: int = 30, include_double: bool = False) -> str:
    """Generate header file footer with dispatcher(s)."""
    footer = f'''
// =============================================================================
// Float Dispatcher
// =============================================================================

static inline int fht_neon_{VERSION}_float(float *buf, int log_n) {{
    if (log_n < 0 || log_n > {max_log_n}) return 1;
    if (log_n == 0) return 0;
    switch (log_n) {{
'''
    for i in range(1, max_log_n + 1):
        footer += f'        case {i}: fht_neon_{VERSION}_float_{i}(buf); break;\n'

    footer += f'''    }}
    return 0;
}}
'''

    if include_double:
        footer += f'''
// =============================================================================
// Double Dispatcher
// =============================================================================

static inline int fht_neon_{VERSION}_double(double *buf, int log_n) {{
    if (log_n < 0 || log_n > {max_log_n}) return 1;
    if (log_n == 0) return 0;
    switch (log_n) {{
'''
        for i in range(1, max_log_n + 1):
            footer += f'        case {i}: fht_neon_{VERSION}_double_{i}(buf); break;\n'

        footer += f'''    }}
    return 0;
}}
'''

    footer += f'''
#endif // FHT_NEON_{VERSION.upper()}_H
'''
    return footer


def generate_full_header(kernels_by_logn: dict, max_log_n: int = 30,
                         double_kernels_by_logn: Optional[dict] = None) -> str:
    """Generate complete header file with all kernels."""
    code = generate_header_start()

    # Float kernels
    code += '// =============================================================================\n'
    code += '// Float Kernels\n'
    code += '// =============================================================================\n\n'

    for log_n in range(1, max_log_n + 1):
        if log_n in kernels_by_logn:
            code += kernels_by_logn[log_n] + '\n'
        else:
            # Use default params
            params = KernelParams()
            code += generate_kernel_v7(log_n, params) + '\n'

    # Double kernels if provided
    include_double = double_kernels_by_logn is not None
    if include_double:
        code += '// =============================================================================\n'
        code += '// Double Kernels\n'
        code += '// =============================================================================\n\n'

        for log_n in range(1, max_log_n + 1):
            if log_n in double_kernels_by_logn:
                code += double_kernels_by_logn[log_n] + '\n'
            else:
                # Use default params
                params = KernelParams()
                code += generate_kernel_double_v7(log_n, params) + '\n'

    code += generate_header_end(max_log_n, include_double=include_double)
    return code


# =============================================================================
# Test / Main
# =============================================================================

def main():
    """Test the V7 generator."""
    print("=" * 70)
    print("Testing V7 Generator - Float")
    print("=" * 70)

    test_sizes = [4, 8, 10, 14, 18, 22]

    for log_n in test_sizes:
        print(f"\nlog_n = {log_n}:")

        # Test different parameter combinations
        configs = [
            KernelParams(strategy='iterative', radix=2, unroll_factor=1),
            KernelParams(strategy='iterative', radix=4, unroll_factor=1),
            KernelParams(strategy='iterative', radix=2, unroll_factor=4),
            KernelParams(strategy='recursive', threshold=min(10, log_n), radix=2),
            KernelParams(strategy='recursive', threshold=min(10, log_n), radix=4, unroll_factor=2),
        ]

        for params in configs:
            try:
                code = generate_kernel_v7(log_n, params)
                print(f"  {params.short_desc():40s}: {len(code):6d} chars")
            except Exception as e:
                print(f"  {params.short_desc():40s}: FAILED - {e}")

    # Test double precision
    print("\n" + "=" * 70)
    print("Testing V7 Generator - Double")
    print("=" * 70)

    for log_n in test_sizes:
        print(f"\nlog_n = {log_n}:")

        # Test different parameter combinations (radix always forced to 2)
        configs = [
            KernelParams(strategy='iterative', radix=2, unroll_factor=1),
            KernelParams(strategy='iterative', radix=2, unroll_factor=4),
            KernelParams(strategy='recursive', threshold=min(10, log_n), radix=2),
            KernelParams(strategy='recursive', threshold=min(10, log_n), radix=2, unroll_factor=2),
        ]

        for params in configs:
            try:
                code = generate_kernel_double_v7(log_n, params)
                print(f"  {params.short_desc():40s}: {len(code):6d} chars")
            except Exception as e:
                print(f"  {params.short_desc():40s}: FAILED - {e}")

    # Generate sample header with both float and double
    print("\n" + "=" * 70)
    print("Generating sample header with float and double kernels...")

    float_kernels = {}
    double_kernels = {}
    default_params = KernelParams(strategy='recursive', threshold=14, radix=4, unroll_factor=2)

    for log_n in range(1, 27):
        params = default_params
        if log_n <= 14:
            params = KernelParams(strategy='iterative', radix=4, unroll_factor=2)

        try:
            float_kernels[log_n] = generate_kernel_v7(log_n, params)
        except:
            float_kernels[log_n] = generate_kernel_v7(log_n, KernelParams())

        # Double uses radix-2 (forced internally)
        double_params = KernelParams(strategy=params.strategy, threshold=params.threshold,
                                     radix=2, unroll_factor=params.unroll_factor)
        try:
            double_kernels[log_n] = generate_kernel_double_v7(log_n, double_params)
        except:
            double_kernels[log_n] = generate_kernel_double_v7(log_n, KernelParams())

    header = generate_full_header(float_kernels, max_log_n=26,
                                  double_kernels_by_logn=double_kernels)

    with open('fht_neon_v7_sample.h', 'w') as f:
        f.write(header)

    print(f"Wrote fht_neon_v7_sample.h ({len(header)} chars)")


if __name__ == '__main__':
    main()
