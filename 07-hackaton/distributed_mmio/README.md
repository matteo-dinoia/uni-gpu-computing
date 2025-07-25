# Distributed MMIO

Lightweight Templated `C++` library for local or distributed reading of Matrix Market files.

This repository integrates with MtxMan ([https://github.com/ThomasPasquali/MtxMan](https://github.com/ThomasPasquali/MtxMan)) to simplify the management of Matrix Market files, check it out!

## Including "Distributed MMIO" in your project

Copy the repo into your project or clone it as a git submodule.

Then simply add it to you CMake:

```cmake
add_subdirectory(distributed_mmio)
```

If you are not using CMake, make sure to include the `distributed_mmio/include` directory and `distributed_mmio/src/mmio.cpp`, `distributed_mmio/src/mmio_utils.cpp` source files.

<!-- ## Including "Distributed MMIO" with CMake

Simply add to your `CMakeLists.txt` the following:

```cmake
include(FetchContent)

FetchContent_Declare(
  distributed_mmio
  GIT_REPOSITORY https://github.com/HicrestLaboratory/distributed_mmio.git
  GIT_TAG        main # or a specific tag/commit
)

FetchContent_MakeAvailable(distributed_mmio)

target_link_libraries(my_target PRIVATE distributed_mmio)
``` -->

## Usage Examples

### Non-distributed Matrix Market File CSR Read 

```c++
#include "../distributed_mmio/include/mmio.h"
// ...
CSR_local<uint32_t, float> *csr_matrix = Distr_MMIO_CSR_local_read<uint32_t, float>("path/to/mtx_file");
COO_local<uint64_t, double> *coo_matrix = Distr_MMIO_COO_local_read<uint64_t, double>("path/to/mtx_file");
```

Explicit template instantiation is currently available for types:

| Index Type | Value Type |
|------------|------------|
| uint32_t   | float      |
| uint32_t   | double     |
| uint64_t   | float      |
| uint64_t   | double     |

> If you need other, add the declaration at the end of `mmio.cpp`. 

# Binary Matrix Market (.bmtx)

This repository also allows to convert, read and write matrices into a binary format.

> **IMPORTANT.** `distributed_mmio` recognizes a file as Binary Matrix Market ONLY if its file extension is `.bmtx`. Currently there are no explicit flags to override this behaviour.

## How it works

The idea is quite simple. A `BMTX` file is formatted as follows:

```
%%MatrixMarket <original header entries> <indices bytes> <values bytes>   // Added two custom values to the header
% <original multiline custom header>
...
% <original multiline custom header>
<n rows> <n cols> <n entries>                                             // As in the original MM format
<
  triples or couples in binary
  (types sizes accordigly with indices bytes and values bytes)
>
```

## mtx_to_bmtx Converter

CMake has a target named `mtx_to_bmtx` which compiles the converter. Once compiled:

```bash
build/mtx_to_bmtx path/to/.mtx  # Converts an MTX file to BMTX. Values (if present) will be written using 4 bytes (float)
build/mtx_to_bmtx path/to/.bmtx # Converts an BMTX file to MTX

build/mtx_to_bmtx path/to/.mtx [-d|--double-val] # Converts an MTX file to BMTX using 8 bytes for values (double)
```

> **NOTE** The size of indices selected automatically in order to maximize compression while mantaining integrity.