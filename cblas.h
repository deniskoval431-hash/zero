#ifndef LITLM_CBLAS_H
#define LITLM_CBLAS_H

/*
 * Minimal CBLAS declaration surface for literary_lm.c.
 * Only cblas_sgemm is used; the symbols are provided by libopenblas.dll
 * shipped alongside the repository.  Matches the standard cblas header
 * signature (netlib CBLAS).
 */

#include <stddef.h>

typedef enum CBLAS_ORDER       { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE   { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;

void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa,
                 const CBLAS_TRANSPOSE transb, const ptrdiff_t M,
                 const ptrdiff_t N, const ptrdiff_t K, const float alpha,
                 const float *a, const ptrdiff_t lda, const float *b,
                 const ptrdiff_t ldb, const float beta, float *c,
                 const ptrdiff_t ldc);

#endif /* LITLM_CBLAS_H */
