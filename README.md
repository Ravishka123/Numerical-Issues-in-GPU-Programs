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
- CuPy does not support dtype=object arrays which is supported by NumPy. [22]

## METHODOLOGY

In order to collect samples of GPU programming related numerical issues, we retrieve commit samples from GitHub repositories. First, we obtain a list of GitHub issues mentioning keywords associated with numerical computation using the GitHub search API. Next, we searched issue titles and descriptions for the following keywords *’nan’, ’overflow’, ’underflow’, ’infinity’, ’infinite’, ’precision’, ’unstable’, ’instability’, ’ringing’, ’unbounded’, ’roundoff ’, ’truncation’, ’rounding’, ’diverge’, ’cancellation’, ’cancel’, ’accuracy’, ’accurate’* to find numerical issues related to GPU programming. This is a similar approach followed by Franco et al. [23] who conducted a study on numerical bugs in numerical libraries in CPU computing. We also added extra keywords such as *’numeric’, ’floating’, ’floats’, ’fp’, ’cufft’, ’cublas’, ’curand’, ’cuda math library’, ’cusolver’, ’cusparse’, ’cutensor’, ’GPU’, and, ’GPU Programming’* to verify numerical issu es related to GPU programs. The libraries such as CuFFT [10], Cublas [9], CuDNN [5], Math library [8], cuRand [1], and cuSolver [11] are GPU specific math libraries. Therefore, using a keyword search for these libraries help us to extract numerical issues associated with GPU programs. Our study includes issues related to a library called CuPy [17]. CuPy is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python [17]. CuPy also provides access to low-level CUDA features. Developer can pass ndarray to existing CUDA C/C++ programs via RawKernels, use Streams for performance, or even call CUDA Runtime APIs directly. [17]. CuPy library supports all the CUDA math libraries which improves the speedup of numerical computations in the GPU.

We chose to examine bug reports to have access to more bug information that includes original reports from users and developer discussions. We included bug reports that are (1) confirmed by developers (e.g., status is closed), (2) a bug fix is available (3) more likely to be numerical bugs. We manually examined the selected bugs by reading their descriptions, comments, and any associated commits or pull requests to verify that they were numerical bugs of interest to our study. Then, we made a second pass through all bugs that passed this final selection criterion, and classified them according to the symptoms they displayed and the strategies used to implement their fixes. Table 1 provides the summary of our dataset.

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Table1.JPG)

## Numerical Bug Study 

We found 202 numerical issues in 645 issues inspected and developed a categorization of numerical bugs that
consist of four groups: a*ccuracy, special value, correctness, and environmental*.Table 2 shows the distribution
of each kind of bug. Next we describe the characteristics of each bug category.

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Table2.JPG)

### Accuracy Bugs

We classified bugs as accuracy bugs when precision loss due to rounding or truncation errors led to an incorrect
result. We found 50 accuracy bugs out of 202 issues. We found that accuracy bugs occur due to several reasons:
- Precision issues can cause accuracy bugs when using too much precision, the performance degrade. Also,
using less precision can cause precision loss yielding to incorrect results.
- Roundoff errors or truncation errors can lead to precision loss.
- Floating-point arithmetic can cause underflow and overflows can lead to accuracy bugs.
- Datatype casting can also change the numerical precision of the variable causing accuracy bugs. Casting
issues are mostly seen in mixed precision computations.

An example of accuracy bug due to roundoff issue is seen in the project CuPy [20]. When computing the
results for order=none and 1, the program produces incorrect results. The results are validated by testing with
the SciPy library. The figure 3 shows the code snippet of the issue. The issue occurs because of an rounding
off error of the value in the variable point. The fix for this issue is to change the values of the variable point
as shown in the figure 4.

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure3.JPG)
![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure4.JPG)

Another example of the accuracy bug is seen the project CuPy [18]. The code snippet of the issue is shown in the figure 5. CuPy’s convolve2d behaves differently from SciPy’s convolve2d when using two arrays with different dtypes. The user wanted to test how arrays of different datatypes behave with the convolve2d. When using integers with different precision, CuPy’s convolve2d leads to integer overflow. If the precisions of the convolved integer arrays are different, the calculation is limited by the lowest precision and integer overflow occurs. However, the SciPy library behaves correctly When convolving arrays of mixed dtypes, the calculations should be performed on the highest-precision dtype. The developers has suggested a temporary fix for this issue in the figure 6 by upcasting the array with the lower precision.

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure5.JPG)
![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure6.JPG)

Accuracy bugs produce various symptoms. Most of the accuracy bugs produces wrong results. These results are often tested with the CPU results. Results of CuPy library’s accuracy issues are tested with NumPy or SciPy versions. Some programs produce error messages relavant to the overflow, underflow, or casting issues. Performance of the computations or high amount of memory are other symptoms that the accuracy bugs produce.

**Fixing Accuracy Bugs:** Most common fixing strategy for accuracy bugs is to change the numerical
precision depending on the situation. If the arithmetic uses few precision, the user can switch to high precision.
Another common fixing strategy is to transform the arithmetic expressions to improve their precision. Also,
strategies like up-casting or down-casting can be helpful to fix underflow and overflow issues. Unit testing tools
can be developed to test the input cases for different arithmetic expressions to understand the cases when the
accuracy bugs occur in a program.

### Correctness Bugs
A correctness bugs is caused by any error in the implementation of an algorithm that have to do primarily with its mathematical or algorithmic structure. Correctness bugs are the most type of numerical bugs; 110 out of 202 numerical bugs are correctness bugs. We found that correctness bugs occur because of several reasons:
- Errors in using incorrect datatypes in expressions
- Errors in unsupported datatypes in compiler optimizations
- Divide by zero issue

An example of a correctness bug is seen in the project CuPy [21] shown by the code snippet in the figure 7.The issue with the code snippet is that the user cannot pass the function cupy.mean which leads to a bad type error during the compilation. The fix for this problem is by creating a ReductionKernel by the user to support the cupy.mean that is shown in the figure 8.

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure7.JPG)
![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure8.JPG)

Correctness bugs produce errors such as compile errors, and type errors. The program may crash with an infinite loop. Also, using incorrect expressions and datatype can cause performance issues. Another symptom that a correctness bug can produce is floating point exceptions because of divide by zero issues.

**Fixing Correctness Bugs:** User needs to have a domain specific knowledge to understand the data types
that certain numerical expressions support. Also, updating the library versions can help to fix correctness bugs
because a new version can add the support for new datatypes. Another common strategy is fixing expressions/using correct expressions and avoiding the use of type conversions in certain optimization levels.
Also, CuPy library uses RawKernel, ReductionKernel, and Element-wise Kernel to improve the quality of numerical computations and these stragegies are not supported by the NumPy or SciPy libraries. cupy.RawKernel Easily uses raw CUDA C kernel functions from python.cupy.ReductionKernel defines the own custom reduction kernel operation cupy.ElementwiseKernel defines custom element wise operation kernel. These techniques improve the performance as well as the numerical stability of the CuPy library.

### Special-value bugs
We refer to signed zero (-0/+0), infinities, and NaN (Not a Number) as special values. We categorized an issue in to this category when the program produces any special value outputs, program fails with NaN inputs, or producing NaN during the computation because of the precision loss leading to infinite values. A special-value bug produces wrong results that consist of special values in the output as symptoms. An example of a Special Value bug is seen in the project transformers [28]. The code snippet in the figure 9 shows the part of the code when the developer identifies that the output of the program produces NaNs with fp16 datatype. The reason for the issue is that the tensor layer output contains inf values that leads to produce NaN values in the output.

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure9.JPG)

Another example is seen in the project pytorch [31]. The code snippet shown in the figure 10 when a.grad produces NaN output with FP16. However, the program works correctly with FP32. The reason for this issue is when the number is small enough to generate infinite values. The fix for this issue is shown in the figure 11 by avoiding values that are too low.


![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure10.JPG)
![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure11.JPG)

**Fixing Special Value Bugs:** The common fixing strategy is to add NaN checks which require the knowledge on when to perform a NaN check in the program. Also checking for signed zero values can help to solve the bugs. Another common fixing strategy is to avoid using numbers that are lesser the lowest number for a certain datatype. This prevents from producing infinite values in the program.

###  Environmental issues
An environmental issue can cause a program to fail because of the use of unsupported GPUs for specific computations. Also, double precision computations are extremely slow in GPUs. Therefore, we added this type of bugs that cause because of the GPUs. Bugs in this category produces symptoms related to bad performance. An example is seen in the project CuPy [19]. The code snippet in the figure 12 when the matrix multiplication using NumPy is faster than the CuPy with double precision numbers. The issue cannot be fixed and developers concerened of changing the datatype to FP32 as a temporary fix.

The fixing strategies for environmental issues are either changing the device that can be costly or changing the datatype by checking whether the GPU supports mixed precision computations.

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Figure12.JPG)

## General Symptoms
We identified three common symptoms of numerical bugs: wrong values, crashes, and bad performance. Table 3 shows the categorization of the symptoms. We found that the majority of numerical bugs (102 out of 202) are revealed by program crashes resulting error messages, and infinite loop. This is followed by producing wrong values (72 out of 202), and causing bad performance (28 out of 202).

![This is an image](https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Resources/Table3.JPG)

## CONCLUSION 

This paper presented the comprehensive study of real world numerical bugs in GPU programming. We examined 645 issues from a open-source GPU programs and the CuPy library. We identified and carefully examined 202 numerical bugs. We found that numerical bugs can be largely categorized into four groups: accuracy bugs, special-value bugs, environmental issues, and correctness bugs. We discussed the characteristics of numerical bugs. We found that the most common symptom for numerical bugs are wrong results, followed by crashes and bad performance.

Future work includes examining more numerical bugs in CuPy library because our dataset consist of more
unexplored CuPy related issues.

Dataset: https://github.com/Ravishka123/Numerical-Issues-in-GPU-programs/blob/main/Data/MergedFinalListFP.xlsx


## References
1. curand. https://developer.nvidia.com/curand.
2. Deep Learning Performance Documentation. https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html.
3. Floating Point and IEEE-754 Compliance for NVIDIA GPUs. https://on-demand.gputechconf.com/gtc-express/2011/presentations/floatingpointwebinarjune2011.pdf.
4. Mixed precision nbsp;: nbsp; tensorflow core.
5. NVIDIA cuDNN. https://developer.nvidia.com/cudnn.
6. Floating point operations difference between CPU and GPU, Nov 2012. https://forums.developer.nvidia.com/t/floating-point-operations-difference-between-cpu-and-gpu/27334.
7. Floating point: CPU and GPU Differences, Aug 2013. http://www.cudahandbook.com/2013/08/floating-point-cpu-and-gpu-differences/.
8. Cuda math library, Mar 2019. https://developer.nvidia.com/cuda-math-library.
9. cublas, Apr 2021. https://developer.nvidia.com/cublas.
10. cufft, Apr 2021. {https://developer.nvidia.com/cufft},journalNVIDIA Developer.
11. cusolver, Apr 2021. https://developer.nvidia.com/cusolver.
12. Fermi (microarchitecture), Sep 2021. https://en.wikipedia.org/wiki/Fermi_(microarchitecture).
13. Fma instruction set, Sep 2021. https://en.wikipedia.org/wiki/FMA_instruction_set.
14. Kepler (microarchitecture), Aug 2021. https://en.wikipedia.org/wiki/Kepler_(microarchitecture).
15. List of Nvidia graphics processing units, Sep 2021. https://en.wikipedia.org/wiki/List_of_Nvidia_
graphics_processing_units.
16. Earl T Barr, Thanh Vo, Vu Le, and Zhendong Su. Automatic detection of floating-point exceptions. ACM
Sigplan Notices, 48(1):549–560, 2013.
17. Cupy. Cupy/cupy: Numpy amp; scipy for gpu.
18. Cupy. Cupyx.scipy.signal.convolve2d: Integer overflow when using mixed dtypes · issue 6047 · cupy/cupy.
19. Cupy. [issue] cp.matmul slower than np.matmul ?? · issue 4891 · cupy/cupy.
20. Cupy. Ndimage.map coordinates performs incorrectly with several order values · issue 4550 · cupy/cupy.
21. Cupy. Passing cupy functions into cupyx.scipy.ndimage.filters.generic filter causes a typeerror · issue 3909· cupy/cupy.
22. Cupy. Valueerror when loading .npy using np.load() · issue 3701 · cupy/cupy.
23. Anthony Di Franco, Hui Guo, and Cindy Rubio-Gonz´alez. A comprehensive study of real-world numerical
bug characteristics. In 2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 509–519. IEEE, 2017.
24. Zhoulai Fu and Zhendong Su. Effective floating-point analysis via weak-distance minimization. In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation, pages 439–452, 2019.
25. Geetika Gupta. Difference between single-, double-, multi-, mixed-precision: Nvidia blog, November 2019. https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multiand-mixed-precision-computing/.
26. Nhut-Minh Ho, Himeshi De silva, and Weng-Fai Wong. GRAM: A Framework for Dynamically Mixing Precisions in GPU Applications. ACM Transactions on Architecture and Code Optimization (TACO),18(2):1–24, 2021.
27. Nhut-Minh Ho and Weng-Fai Wong. Exploiting half precision arithmetic in Nvidia GPUs. In 2017 IEEE High Performance Extreme Computing Conference (HPEC), pages 1–7. IEEE, 2017.10
28. Huggingface. T5model in fp16 still yield nan with more complex examples · issue 4586 · huggingface/transformers.
29. Ignacio Laguna. Fpchecker: Detecting floating-point exceptions in gpu applications. In 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 1126–1129. IEEE,2019.
30. Lukas Polok and Pavel Smrz. Increasing double precision throughput on NVIDIA Maxwell GPUs. In
Proceedings of the 24th High Performance Computing Symposium, pages 1–8, 2016.
31. Pytorch. Torch.norm gives nan gradient when i input small-value float16 tensor · issue 43211 · pytorch/pytorch.
32. Royi, Shehzan Mohammed, Pavan Yalamanchili, Micha l Janiszewski, Gurga, Dan, XXXeeqXXX, and Don Karam. Explaining fp64 performance on gpus, Jun 2015. https://arrayfire.com/
explaining-fp64-performance-on-gpus/.
33. Grigory Sapunov. Fp64, fp32, fp16, bfloat16, tf32, and other members of the zoo, May 2020.
34. Sruthikesh Surineni, Ruidong Gu, Huyen Nguyen, and Michela Becchi. Understanding the performanceaccuracy tradeoffs of floating-point arithmetic on gpus. In 2017 IEEE International Symposium on Workload Characterization (IISWC), pages 207–218. IEEE, 2017.
35. Enyi Tang, Earl Barr, Xuandong Li, and Zhendong Su. Perturbing numerical calculations for statistical analysis of floating-point program (in) stability. In Proceedings of the 19th international symposium on Software testing and analysis, pages 131–142, 2010.
36. Tensorflow. Tensorflow 2.2 using tf.float16 executes only on cpu · issue 41783 · tensorflow/tensorflow.
37. Tensorflow. [tf lite] tflitegpudelegate init: Conv 2d: Unsupported data type for float32 tensor · issue 40357· tensorflow/tensorflow.
38. Ran Wang, Daming Zou, Xinrui He, Yingfei Xiong, Lu Zhang, and Gang Huang. Detecting and Fixing Precision-Specific Operations for Measuring Floating-Point Errors. In Proceedings of the 2016 24th ACM SIGSOFT International Symposium on Foundations of Software Engineering, pages 619–630, 2016.
39. Nathan Whitehead and Alex Fit-Florea. Precision & performance: Floating point and IEEE 754 compliance
for NVIDIA GPUs. rn (A+ B), 21(1):18749–19424, 2011.
40. Daming Zou, Ran Wang, Yingfei Xiong, Lu Zhang, Zhendong Su, and Hong Mei. A genetic algorithm for detecting significant floating-point inaccuracies. In 2015 IEEE/ACM 37th IEEE International Conference on Software Engineering, volume 1, pages 529–539. IEEE, 2015.
41. Daming Zou, Muhan Zeng, Yingfei Xiong, Zhoulai Fu, Lu Zhang, and Zhendong Su. Detecting floatingpoint errors via atomic conditions. Proceedings of the ACM on Programming Languages, 4(POPL):1–27, 2019.







