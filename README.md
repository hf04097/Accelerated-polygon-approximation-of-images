# Accelerated Polygon Approximation Of Images
Polygon approximation is an interesting application and lends itself to uses in several domains. This approximation is primarily used for compression as complex images and figures can be represented in significantly less amount of data and require much less computation while preserving the essential information about the curve and the image.

Using a reference image as input, polygons are used to approximate the image in the YCrCb color space using genetic algorithms.

Several approaches were used to parallelize the application and significant speedup in application time was obtained. The different implemented techniques include

* Serial implementation
* Cuda naive implementation
* CUDA Implementation with Shared Memory
* Implementation with CUDA Code Optimizations

# How to run
Import any of the implementations and call the `run_iterations` method.

## Example
```
from Optimized1_GPU_RGB import *
run_iterations([2, 50], [15], validation = True)
```
This would run the algorithm for generation 2 and 50 seperately, for a population size of 15. `validation = True` might reduce the runtime, as it compares each value computed on GPU with CPU.

For more examples, visit the example [Google Colab Notebook](https://colab.research.google.com/drive/10zzH5kMh8ikfoeHhYjRA_Rgb_9trRzUZ?usp=sharing)
