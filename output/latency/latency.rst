.. list-table:: PolarFire VectorBlox latency on 192.168.20.6 (50 images per network)
   :header-rows: 1
   :widths: 44 28 28

   * - Metric
     - MobilePose 224 × 224
     - FCOS 384 × 288
   * - Images
     - 50
     - 50
   * - Loops per image
     - 1
     - 1
   * - Accelerator mean (ms)
     - 17.20
     - 394.67
   * - Accelerator std (ms)
     - 0.01
     - 0.02
   * - Accelerator median (ms)
     - 17.20
     - 394.67
   * - Accelerator min (ms)
     - 17.18
     - 394.64
   * - Accelerator max (ms)
     - 17.21
     - 394.72
   * - Dequantize mean (ms)
     - 4.33
     - 1.81
   * - Dequantize std (ms)
     - 0.05
     - 0.05
   * - ONNX Runtime mean (ms)
     - 16.80
     - 13.37
   * - ONNX Runtime std (ms)
     - 0.12
     - 0.52
   * - CPU post-processing mean (ms)
     - 21.13
     - 15.18
   * - CPU post-processing std (ms)
     - 0.14
     - 0.52
   * - End-to-end mean (ms)
     - 38.33
     - 409.85
   * - End-to-end std (ms)
     - 0.14
     - 0.52
   * - End-to-end median (ms)
     - 38.31
     - 409.91
   * - Accelerator throughput (FPS)
     - 58.15
     - 2.53
   * - End-to-end throughput (FPS)
     - 26.09
     - 2.44
