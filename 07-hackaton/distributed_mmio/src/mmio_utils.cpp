#include <stdio.h>
#include <string>
#include <vector>

#include "../include/mmio.h"
#include "../include/mmio_utils.h"

#define MMIO_UTILS_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template void print_csr(CSR_local<IT, VT> *csr, std::string header); \
  template void print_csr_as_dense<IT, VT>(CSR_local<IT, VT> *csr, std::string header); \
  template void print_coo(COO_local<IT, VT> *coo, std::string header); \

template<typename IT, typename VT>
void print_csr(CSR_local<IT, VT> *csr, std::string header) {
  if (header != "") {
    printf("%s -- ", header.c_str());
  }
  std::string I_FMT = "%3u";
  if constexpr (std::is_same<IT, uint64_t>::value)
    I_FMT = "%3lu";

  char fmt[100];
  snprintf(fmt, 100, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
  printf(fmt, csr->nrows, csr->ncols, csr->nnz);

  snprintf(fmt, 100, "%s ", I_FMT.c_str());
  printf("idx   : ");
  for (IT i = 0; i < csr->nnz; ++i) printf(fmt, i);
  printf("\nrowptr: ");
  for (IT i = 0; i <= csr->nrows; ++i) printf(fmt, csr->row_ptr[i]);
  printf("\ncolidx: ");
  for (IT i = 0; i < csr->nnz; ++i) printf(fmt, csr->col_idx[i]);
  if (csr->val != NULL) {
    printf("\nval:    ");
    for (IT i = 0; i < csr->nnz; ++i) {
      printf("%.1f ", csr->val[i]); // TODO handle different VT
    }
  }
  printf("\n");
}

template<typename IT, typename VT>
void print_csr_as_dense(CSR_local<IT, VT> *csr, std::string header) {
  std::vector<std::vector<VT>> dense_matrix(csr->nrows, std::vector<VT>(csr->ncols, 0.0f));
  for (IT row = 0; row < csr->nrows; ++row) {
    for (IT idx = csr->row_ptr[row]; idx < csr->row_ptr[row + 1]; ++idx) {
      IT col = csr->col_idx[idx];
      dense_matrix[row][col] = csr->val != NULL ? csr->val[idx] : 1.0f; // TODO handle different VT
    }
  }
  if (header != "") {
      printf("%s -- ", header.c_str());
  }

  std::string I_FMT = "%3u";
  if constexpr (std::is_same<IT, uint64_t>::value)
    I_FMT = "%3lu";

  char fmt[100];
  snprintf(fmt, 100, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
  printf(fmt, csr->nrows, csr->ncols, csr->nnz);
  
  for (IT row = 0; row < csr->nrows; ++row) {
    for (IT col = 0; col < csr->ncols; ++col) {
      if (dense_matrix[row][col] == 0)
        printf("   - ");
      else
        printf("%4.0f ", dense_matrix[row][col]); // TODO handle different VT
    }
    printf("\n");
  }
}

template<typename IT, typename VT>
void print_coo(COO_local<IT, VT> *coo, std::string header) {
  if (header != "") {
    printf("%s -- ", header.c_str());
  }
  
  std::string I_FMT = "%4u";
  if constexpr (std::is_same<IT, uint64_t>::value)
    I_FMT = "%4lu";

  char fmt[100];
  snprintf(fmt, 100, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
  printf(fmt, coo->nrows, coo->ncols, coo->nnz);

  snprintf(fmt, 100, "%s ", I_FMT.c_str());
  printf("idx: ");
  for (IT i = 0; i < coo->nnz; ++i) printf(fmt, i);
  printf("\nrow: ");
  for (IT i = 0; i < coo->nnz; ++i) printf(fmt, coo->row[i]);
  printf("\ncol: ");
  for (IT i = 0; i < coo->nnz; ++i) printf(fmt, coo->col[i]);
  if (coo->val != NULL) {
    printf("\nval: ");
    for (IT i = 0; i < coo->nnz; ++i) {
      printf("%4.1f ", coo->val[i]); // TODO handle different VT
    }
  }
  printf("\n");
}


MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint32_t, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint32_t, double)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, double)