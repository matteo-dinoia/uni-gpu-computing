#include <stdio.h>
#include <string>
#include <algorithm>
#include <chrono>

#include "../include/mmio.h"
#include "../include/mmio_utils.h" // FIXME

#define BRIGHT_CYAN     "\033[96m"
#define RESET           "\033[0m"

#define CPU_TIMER_DEF(name) \
  std::chrono::high_resolution_clock::time_point __timer_start_##name, __timer_stop_##name;

#define CPU_TIMER_START(name) \
  __timer_start_##name = std::chrono::high_resolution_clock::now();

#define CPU_TIMER_STOP(name) \
  __timer_stop_##name = std::chrono::high_resolution_clock::now();

#define CPU_TIMER_ELAPSED(name) \
  (std::chrono::duration<float>(__timer_stop_##name - __timer_start_##name).count()*1e3)

#define CPU_TIMER_PRINT(name) \
  printf(BRIGHT_CYAN "Timer [%s] elapsed: %f ms\n" RESET, #name, CPU_TIMER_ELAPSED(name));

#define CPU_TIMER_INIT(name) CPU_TIMER_DEF(name) CPU_TIMER_START(name)

#define CPU_TIMER_CLOSE(name) CPU_TIMER_STOP(name) CPU_TIMER_PRINT(name)

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    // printf("Usage: %s <filename> [-r|--reverse] [-d|--double-val]\n", argv[0]);
    printf("Usage: %s <filename> [-d|--double-val]\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::string filename = argv[1];
  // bool reverse = false;
  bool double_val = false;

  uint32_t arg_i = 2;
  while (arg_i < argc) {    
    std::string flag = argv[arg_i];
    // if (flag == "-r" || flag == "--reverse") {
    //   reverse = true;
    // } else
    if (flag == "-d" || flag == "--double-val") {
      double_val = true;
    } else {
      printf("Unknown option: %s\n", argv[arg_i]);
    }
  }

  Matrix_Metadata mtx_meta;
  mtx_meta.val_bytes = double_val ? 8 : 4;
  CPU_TIMER_INIT(COO_read)
  COO_local<uint64_t, double> *coo = Distr_MMIO_COO_local_read<uint64_t, double>(filename.c_str(), false, &mtx_meta);
  CPU_TIMER_CLOSE(COO_read)
  if (coo == NULL) {
    fprintf(stderr, "Something went wrong\n");
    exit(EXIT_FAILURE);
  }

  std::string out_filename = filename;
  size_t last_dot = out_filename.find_last_of('.');
  if (last_dot != std::string::npos) {
    out_filename = out_filename.substr(0, last_dot);
  }

  bool converting_to_bmtx = !is_file_extension_bmtx(filename);

  // print_coo(coo);
  // if (!converting_to_bmtx) exit(0);

  CPU_TIMER_INIT(Conversion)
  if (converting_to_bmtx) {
    printf("Converting MTX file to BMTX...\n");
    out_filename += ".bmtx";
    Distr_MMIO_COO_local_write(coo, out_filename.c_str(), true, &mtx_meta);
    printf("BMTX file written to %s\n", out_filename.c_str());
  } else {
    printf("Converting BMTX file to MTX...\n");
    out_filename += ".mtx";
    Distr_MMIO_COO_local_write(coo, out_filename.c_str(), false, &mtx_meta);
    printf("MTX file written to %s\n", out_filename.c_str());
  }
  CPU_TIMER_CLOSE(Conversion)

  Distr_MMIO_COO_local_destroy(&coo);

  // Compare file sizes of input and output files
  FILE *f_in = fopen(filename.c_str(), "rb");
  FILE *f_out = fopen(out_filename.c_str(), "rb");
  if (!f_in || !f_out) {
    fprintf(stderr, "Error opening files for size comparison\n");
    if (f_in) fclose(f_in);
    if (f_out) fclose(f_out);
    exit(EXIT_FAILURE);
  }
  fseek(f_in, 0, SEEK_END);
  fseek(f_out, 0, SEEK_END);
  long size_in = ftell(f_in);
  long size_out = ftell(f_out);
  fclose(f_in);
  fclose(f_out);

  printf("Size of input  file (%s): %.3f MB\n", filename.c_str(), size_in/1000000.0f);
  printf("Size of output file (%s): %.3f MB\n", out_filename.c_str(), size_out/1000000.0f);
  printf("Ratio: %.2f%%\n", (float)size_out/size_in*100.0f);

  return 0;
}
