#ifndef NVDR_WASM_MATH_H
#define NVDR_WASM_MATH_H
/* WebAssembly has these as instructions. */
#define sqrt(x)  __builtin_sqrt(x)
#define fabs(x)  __builtin_fabs(x)
#define floor(x) __builtin_floor(x)
#define ceil(x)  __builtin_ceil(x)
/* Only the encoder uses these; declared so it compiles, dropped at link. */
double log(double); double log2(double); double log10(double); double exp(double);
double cos(double); double pow(double, double); long lround(double);
#endif
