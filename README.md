# A Comprehensive Study on Numerical Issues in GPU Programs

##INTRODUCTION

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
