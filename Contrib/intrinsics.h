#pragma once

// Use SSE on intel processors, or use Neon on ARM architectures:
#ifdef __ARM_NEON__
#include <arm_neon.h>
typedef float32x4_t v4sf; // vector of 4 float

// ARM NEON helper functions, see here: https://pmeerw.net/blog/programming/neon1.html

static __inline v4sf v4sf_mul(v4sf x, v4sf y)
{ return (x * y); }

static __inline v4sf v4sf_and(v4sf x, v4sf y)
{ return vreinterpretq_f32_u32(vreinterpretq_u32_f32(x) & vreinterpretq_u32_f32(y)); }

static __inline v4sf v4sf_andn(v4sf x, v4sf y)
{ return vreinterpretq_f32_u32((~vreinterpretq_u32_f32(x)) & vreinterpretq_u32_f32(y)); }

static __inline v4sf v4sf_or(v4sf x, v4sf y)
{ return vreinterpretq_f32_u32(vreinterpretq_u32_f32(x) | vreinterpretq_u32_f32(y)); }

static __inline v4sf v4sf_invsqrtv(v4sf x)
{
  v4sf sqrt_reciprocal = vrsqrteq_f32(x);
  return vrsqrtsq_f32(x * sqrt_reciprocal, sqrt_reciprocal) * sqrt_reciprocal;
}

static __inline v4sf v4sf_sqrt(v4sf x) { return x * v4sf_invsqrtv(x); }

static __inline v4sf v4sf_invv(v4sf x)
{
  v4sf reciprocal = vrecpeq_f32(x);
  reciprocal = vrecpsq_f32(x, reciprocal) * reciprocal;
  return reciprocal;
}

static __inline v4sf v4sf_div(v4sf x, v4sf y)
{ 
  v4sf reciprocal = vrecpeq_f32(y);
  reciprocal = vrecpsq_f32(y, reciprocal) * reciprocal;
  return x * v4sf_invv(y);
}
  
static __inline v4sf v4sf_min(v4sf x, v4sf y) { return vminq_f32(x, y); }
static __inline v4sf v4sf_max(v4sf x, v4sf y) { return vmaxq_f32(x, y); }


#else

// Assume intel and SSE when NEON is not true:
#include <xmmintrin.h>
typedef __v4sf v4sf;

static __inline v4sf v4sf_mul(v4sf x, v4sf y) { return __builtin_ia32_mulps(x, y); }
static __inline v4sf v4sf_and(v4sf x, v4sf y) { return __builtin_ia32_andps(x, y); }
static __inline v4sf v4sf_andn(v4sf x, v4sf y) { return __builtin_ia32_andnps(x, y); }
static __inline v4sf v4sf_or(v4sf x, v4sf y) { return __builtin_ia32_orps(x, y); }
static __inline v4sf v4sf_sqrt(v4sf x) { return __builtin_ia32_sqrtps(x); }
static __inline v4sf v4sf_div(v4sf x, v4sf y) { return __builtin_ia32_divps(x, y); }
static __inline v4sf v4sf_min(v4sf x, v4sf y) { return __builtin_ia32_minps(x, y); }
static __inline v4sf v4sf_max(v4sf x, v4sf y) { return __builtin_ia32_maxps(x, y); }


#endif
