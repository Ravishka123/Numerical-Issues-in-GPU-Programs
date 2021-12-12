# A Comprehensive Study on Numerical Issues in GPU Programs

## INTRODUCTION

Numerical applications relies on floating point arithmetic. Floating point numbers are an approximation of real
numbers that follows the IEEE 754 standard floating point format for half, single, and double precision levels.
Applications from several fields including scientific computing, machine learning, earth sciences, graphics, engineering, and finance widely use the floating point arithmetic. However, developers and researchers consider
about the speed up of the applications. Therefore, general purpose graphics processing unit (GPGPU) computing has become a major parallel computing paradigm which supports data parallelism. The data parallelism
helps the programs to execute a single instruction stream on many data elements at the same time. Therefore,
GPUs have the ability to perform a massive number of floating-point calculations per second which improves the
speed up and the efficiency of the programs compared to the CPU programs due to the massive multi-threaded
nature of the GPU hardware.

The general-purpose GPU programming facilitates extraordinary computing power because of the amount
of parallelism available in GPUs to execute applications quickly. GPUs facilitate the ability to compute floating
point operations and produce significant results in a short period of time. Therefore, the capabilities of NVIDIA
GPUs [15] have been expanded in each hardware generation from only supporting single precision in early
NVIDIA architectures, to fully supporting IEEE half, single, double precision, and including FMA (FusedMultiply-Add) [13] operations in modern generations such as Fermi [12] and Kepler [14] GPUs. According to
the evolution of CUDA [3], compute capability below 1.2 only supports single precision floating point arithmetic,
compute capability 1.3 supports both single and double precision arithmetic, and offers double-precision FMA
operations, and compute capability 2.0 and above fully support IEEE compliance. At present, GPUs support
both mixed precision computing [25]. Multi-precision computing uses processors that are capable of calculating
at different precision such as, using double precision when needed, and relying on half- or single-precision
arithmetic for other parts of the application. Mixed-precision computing uses different precision levels within a
single operation to achieve computational efficiency without sacrificing accuracy [25].

Even though, the traditional CPU programming and GPU programming both follows the standard IEEE 754 format for floating point computations, there are differences in terms of floating point computation, accuracy, and performance of the computations between the CPU and the GPU. First, CPUs tend to do floating point calculations in 80-bit ‘extended’ mode while GPUs use 32 bit and 64 bits for single and double precision respectively. [6]. Therefore, different architectures such as x64 and x86, support different levels of precision and rounding under different conditions such as compiler flags [7]. The figure 2 provides some CUDA compiler intrinsics on floating point arithmetic computation. The GPUs support all four rounding modes defined by the IEEE 754. According to Whitehead et al. [39], NVIDIA GPUs differ from the x86 architecture in that rounding modes are encoded within each floating point instruction instead of dynamically using a floating point control word. Exception handlers for floating point exceptions are not supported. There is no status flag to indicate when calculations have overflowed, underflowed, or have involved inexact arithmetic on the GPU. Also, GPUs may produce slightly different results due to the multi threaded environment when the floating point operations run in different orders from one iterations to the other. The race conditions occur during the execution of GPUs is a reason to produce different results. However, the performance of GPU programs degrades when using higher precision such as double values. The CPU to GPU memory transfers and GPU memory allocations take longer time to allocate higher precision numbers. An instances of this issue is addressed in the NVIDIA’s deep learning documentation [2] when half precision (FP16) data compared to higher precision such as single precision (FP32) and double precision (FP64) reduces memory usage of the neural network, allowing training and deployment of larger networks, and FP16 data transfers take less time than FP32 or FP64 transfers in GPU programs. However, CPU programs do not encounter such performance issues. On the other hand, single precision floating point computing, 1) uses less memory, therefore, the data can be transferred into the device faster 2) has less accuracy, therefore, approximations can be used for faster calculations compared to double precision computing. According to Shehzan et al. [32] there is no perfect solution to the problem of choosing between FP32 and FP64 in GPU computing and applications like physics modelling and simulation, high accuracy financial computations etc which call for double precision accuracy at high performance require capable FP64 cards while applications such as image and video processing, signal processing, statistics may not require such high precision and can get away with high FP32 performance only. Therefore, the limitation on floating point arithmetic, and current issues on improving the memory transfers, memory allocations, compiler flags, and performance in the GPU need more addressing to improve the application’s productivity.

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure1.JPG)

Addressing numerical issues in GPU programming is important. Applications that involves double precision
computations may encounter performance issues due to slow memory copies and large memory allocations.
Also, accuracy is an important factor in large-scale computations when data race conditions produce different
results. Therefore, we explore numerical issues mainly on floating point issues and current limitation on GPU
programs.Our dataset consist of 202 numerical issues on GPU programs including the issues associated with
the CuPy library [17]. In addition, users of GPU math libraries such as CuFFT [10], Cublas [9], CuDNN [5],
Math library [8], cuRand [1], and cuSolver [11] find it informative to learn how NVIDIA handles floating point
operations on these libraries with CuPy.The purpose of our study is to explore the GPU program numerical
issues to understand the common issues that developers face while using floating point/numeric operations in
the GPU. We propose to categorize numerical bugs into accuracy bugs, special-value bugs, Environmental issues,
and correctness bugs. We discuss their characteristics, including common symptoms and fixing strategies, and
present real-world examples.

Therefore, we focus on the following questions to expand our study.
**RQ1: How can we group numerical issues into categories that share common causes and
patterns, and how frequently do bugs in each category occur?
RQ2: What are the fixing strategies for each categorized numerical issues?
RQ3: What are the symptoms that these numerical issues produce?**
Our work can improve the programmer’s understanding of the numerical support offered by modern GPUs
and the performance/accuracy trade offs related to the use of different floating-point precision on GPUs. This
study can provide insights to developing automated tools to identify GPU program issues related to floating
point operations.

## Numerical Issues in GPUs

GPU programs encounter several numerical issues. The following are the summary of the GPU specific numerical
issues we found through our study.

- NVIDIA GPUs can run operations in float16 faster than in float32.Variables and a few computations
should still be in float32 for numeric reasons so that the model trains to the same quality. Float16 for
matrix multiplications, convolutions etc. [36]
- Mixed precision policy applies that float16 is used for computations and keep variables in float32. The
issues such as underflow, overflow, and datatype casting issues can occur. [4]
- Older GPUs offer no math performance benefit for using mixed precision. Examples of GPUs that will
benefit most from mixed precision include RTX GPUs, the V100, and the A100. FP32 always uses CUDA
cores. FP16 may or may not use tensor cores. [4]
- Integer quantization is mainly for CPU since CPU calculates integer better than float number. When
you run it on GPU, it converts integer to float internally. [37] Issues such as high memory use due to the
allocation of large memory, and precision loss can occur due to integer behavior in the GPUs.
- Inconsistent handling of NaNs on Cuda compared to CPU tensors.
- IEEE half-precision 16-bit float uses 5 bits exponent, and 10 bits fraction, however, bfloat16 uses 8 bits
exponent,and 7 bits fraction. Precision loss can occur when casting float16 to bfloat16 in mixed precision
computations.
- The NVIDIA GPU architecture uses a little-endian representation. NumPy supports little- and bigendianness. CuPy arrays only use little-endian arrays. Big-endian arrays lead to wrong results.
- Low double precision performance of the GPUs.
- CuPy does not support dtype=object arrays which is supported by NumPy. [22
