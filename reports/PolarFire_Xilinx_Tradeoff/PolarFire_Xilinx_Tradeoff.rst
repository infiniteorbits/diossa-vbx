PolarFire and Xilinx accelerator trade-off
==========================================

.. table::
   :width: 100%
   :widths: auto

   +------------------------------------------------------------------+
   | Date created: 05/10/2026                                         |
   +------------------------------------------------------------------+
   | **Status:** DRAFT                                                |
   |                                                                  |
   | PolarFire measurements are not available yet. Cells marked       |
   | ``TBD`` are filled when the current inference campaign finishes. |
   +------------------------------------------------------------------+
   | **Distribution:**                                                |
   |                                                                  |
   | - LMO (internal),                                                |
   | - European Space Agency (ESA)                                    |
   +------------------------------------------------------------------+
   | **Change Log:**                                                  |
   |                                                                  |
   | -  0.1: First draft. Xilinx baseline taken from the CCN1         |
   |    Design Justification File. PolarFire columns left open.       |
   +------------------------------------------------------------------+

.. table::
   :width: 100%
   :widths: auto

   +-----------------+---------------------------+--------------------------+---------------+----------+
   |                 | **Position**              | **Name**                 | **Signature** | **Date** |
   +-----------------+---------------------------+--------------------------+---------------+----------+
   | **Prepared by** | Computer Vision Engineer  | Maciej Zurad             |               |          |
   +-----------------+---------------------------+--------------------------+---------------+----------+
   | **Reviewed by** |                           |                          |               |          |
   +-----------------+---------------------------+--------------------------+---------------+----------+
   | **Approved by** |                           |                          |               |          |
   +-----------------+---------------------------+--------------------------+---------------+----------+

This document is of Luxembourg origin and is Copyright © of LMO Sarl.

Introduction
============

The DIOSSA RPO payload estimates the 6-degree-of-freedom pose of a target
spacecraft from monocular images. The vision chain is two convolutional
networks in series: FCOS for object detection, then MobilePose for
keypoint regression. In the previous phase those networks were trained
in floating point, refined with quantization-aware training, and compiled
onto the Xilinx Deep Learning Processing Unit (DPU) in an UltraScale+
MPSoC through Vitis-AI 3.5.

This note compares that Xilinx baseline with the same networks compiled
for the Microchip PolarFire SoC and its VectorBlox accelerator. The
networks, the weights, and the task metrics stay the same. The item
under test is the accelerator and its compile path.

Objective
---------

Show that the CCN1 networks can move from the Xilinx DPU to PolarFire
without a material loss of task accuracy, and define the measurements
still required before that conclusion is closed.

Scope
-----

The comparison uses the two networks that were embedded in CCN1 and that
are being recompiled for PolarFire:

- Object detection: FCOS, input 384 × 288.
- Keypoint regression: MobilePose, input 224 × 224, 20 keypoints.

A second FCOS resolution, 640 × 480, was trained in CCN1 and is reported
here as context. It is not part of the PolarFire campaign.

Applicable Documents
--------------------

.. table::
   :width: 100%
   :widths: auto

   +----------+---------------------------------------------------------+------------+
   | **AD-n** | **Document Title**                                      | **Issue**  |
   +----------+---------------------------------------------------------+------------+
   | AD-1     | PL23-0005-DJF-0001 i1.1 (DIOSSA-CCN1)                   | 1.1        |
   +----------+---------------------------------------------------------+------------+

Definitions
-----------

.. table::
   :width: 80%
   :widths: auto

   +-------------+--------------------------------------------------------------+
   | **Acronym** | **Meaning**                                                  |
   +-------------+--------------------------------------------------------------+
   | DPU         | Xilinx Deep Learning Processing Unit                         |
   +-------------+--------------------------------------------------------------+
   | EDGE        | The network after compilation, running on the accelerator    |
   +-------------+--------------------------------------------------------------+
   | FLOAT       | The float32 network, evaluated on a GPU                      |
   +-------------+--------------------------------------------------------------+
   | FCOS        | Fully Convolutional One-Stage object detector                |
   +-------------+--------------------------------------------------------------+
   | IoU         | Intersection over union of the predicted and true box        |
   +-------------+--------------------------------------------------------------+
   | mAP         | Mean average precision                                       |
   +-------------+--------------------------------------------------------------+
   | QAT         | Quantization-aware training; int8-constrained weights,       |
   |             | evaluated in the training framework before compilation       |
   +-------------+--------------------------------------------------------------+

What is being compared
======================

Three representations of each network are kept separate, because they
answer different questions.

.. table:: Representations of each network
   :width: 100%
   :widths: auto

   +------------+---------------------------------------------------------------+----------------------------------------------+
   | **Stage**  | **What it is**                                                | **What a difference means**                  |
   +------------+---------------------------------------------------------------+----------------------------------------------+
   | FLOAT      | Best float32 checkpoint from CCN1, run on a GPU               | Training quality. Independent of the FPGA.   |
   +------------+---------------------------------------------------------------+----------------------------------------------+
   | QAT        | Same checkpoint after quantization-aware training,            | Accuracy given up to make the network        |
   |            | still evaluated in PyTorch                                    | representable in int8.                       |
   +------------+---------------------------------------------------------------+----------------------------------------------+
   | EDGE       | The QAT network compiled and executed on the accelerator      | Accuracy and latency of the flight-like      |
   |            | (Xilinx DPU, or PolarFire VectorBlox)                         | implementation.                              |
   +------------+---------------------------------------------------------------+----------------------------------------------+

FLOAT and QAT are properties of the checkpoint. They are identical for
both targets, and they are taken from AD-1. EDGE is the only stage that
can differ between Xilinx and PolarFire.

The PolarFire builds use the CCN1 QAT checkpoints directly:

- FCOS ``ccn1--quare-delf-epoch19``, input ``1 × 3 × 288 × 384``.
- MobilePose ``ccn1--bijou-rasp-epoch17``, input ``1 × 3 × 224 × 224``,
  20 keypoints.

Xilinx baseline
===============

All figures in this section are copied from AD-1. They were measured on
the CCN1 validation set unless noted otherwise. Inference rates quoted
from AD-1 were measured on a laptop RTX 3070, not on the DPU. AD-1 does
not publish a DPU frame rate.

Object detection (FCOS)
-----------------------

FCOS was selected over Faster R-CNN in CCN1. On a 480 × 640 input the
float FCOS reached IoU 0.874 and mAP at IoU 0.75 of 0.956, against
0.853 and 0.901 for Faster R-CNN, at 45.78 FPS against 37.66 FPS on the
RTX 3070, with 32.1 M parameters against 43 M. Faster R-CNN also cannot
be trained with the Vitis-AI quantization-aware procedure, so it was
dropped.

The embedded resolution is 384 × 288. The larger 640 × 480 float model
is more accurate (IoU 0.958 against 0.925) and about 30% slower on the
GPU (46.13 FPS against 65.12 FPS). Quantization-aware training on
640 × 480 could not be run for long enough: the batch size was limited
to 2 and the run lasted 47.7 minutes, finishing at IoU 0.819. The
384 × 288 QAT run used batch size 7 for 11.4 hours and finished at
IoU 0.858. That is the network compiled for the edge.

.. table:: FCOS, 384 × 288. Validation metrics from AD-1.
   :width: 100%
   :widths: auto

   +-------------------------------+---------------------------+---------------------------+
   |                               | **FLOAT**                 | **QAT**                   |
   |                               | licit-weal                | quare-delf                |
   +-------------------------------+---------------------------+---------------------------+
   | Started from                  | COCO                      | licit-weal                |
   +-------------------------------+---------------------------+---------------------------+
   | Input [W × H]                 | 384 × 288                 | 384 × 288                 |
   +-------------------------------+---------------------------+---------------------------+
   | IoU                           | 0.925                     | 0.858                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP                           | 0.858                     | 0.693                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP at IoU > 0.50             | 0.989                     | 0.978                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP at IoU > 0.75             | 0.946                     | 0.850                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP, large (area > 9216 px)   | —                         | 0.746                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP, medium                   | —                         | 0.691                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP, small (area < 1024 px)   | —                         | 0.623                     |
   +-------------------------------+---------------------------+---------------------------+
   | RTX 3070 throughput           | 65.12 FPS                 | not reported              |
   +-------------------------------+---------------------------+---------------------------+

The accepted quantization loss on this network is 0.067 IoU
(0.925 down to 0.858) and 0.096 mAP at IoU 0.75 (0.946 down to 0.850).
Detection at the loose threshold barely moves: mAP at IoU 0.50 goes
from 0.989 to 0.978.

.. table:: FCOS, 640 × 480. Reported for context. Not in the PolarFire campaign.
   :width: 100%
   :widths: auto

   +-------------------------------+---------------------------+---------------------------+
   |                               | **FLOAT**                 | **QAT**                   |
   |                               | saved-aril                | livid-wire                |
   +-------------------------------+---------------------------+---------------------------+
   | Started from                  | COCO                      | saved-aril                |
   +-------------------------------+---------------------------+---------------------------+
   | Input [W × H]                 | 640 × 480                 | 640 × 480                 |
   +-------------------------------+---------------------------+---------------------------+
   | IoU                           | 0.958                     | 0.819                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP                           | 0.939                     | 0.658                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP at IoU > 0.50             | 0.990                     | 0.944                     |
   +-------------------------------+---------------------------+---------------------------+
   | mAP at IoU > 0.75             | 0.988                     | 0.826                     |
   +-------------------------------+---------------------------+---------------------------+
   | RTX 3070 throughput           | 46.13 FPS                 | not reported              |
   +-------------------------------+---------------------------+---------------------------+

The 640 × 480 QAT model loses 0.139 IoU. After that loss it is less
accurate than the 384 × 288 QAT model (0.819 against 0.858). The
resolution that was actually embedded for the shorter approach is the
smaller one.

Keypoint regression (MobilePose)
--------------------------------

The float model selected in CCN1 is ``blest-harl``, initialised from the
DIOSSA-1 MobilePose checkpoint and trained on 20 keypoints. AD-1 states
that raising the input from 224 × 224 to 448 × 448 did not improve
accuracy and did increase training time, and that predicting all 41
keypoints was clearly worse than the 20-keypoint subset. The float
experiment table itself is a figure in AD-1
(``tables/big-kr-experiments-table.png``) and is not in this repository,
so the float row below contains only the two values written in the
prose.

``blest-harl`` is reported at 3.799 L2 pixel error. The prose does not
repeat whether that figure is on the 2590 × 1942 camera image or on the
network input. The QAT section does give the float reference for L1 on
the camera image: 4.783 px, against 5.303 px for the best QAT model.

The QAT model is ``bijou-rasp``, started from ``blest-harl``. It is the
checkpoint compiled for the edge (``ccn1--bijou-rasp-epoch17``).

.. table:: MobilePose, 20 keypoints. Metrics from AD-1.
   :width: 100%
   :widths: auto

   +------------------------------------------------+---------------------------+---------------------------+
   |                                                | **FLOAT**                 | **QAT**                   |
   |                                                | blest-harl                | bijou-rasp                |
   +------------------------------------------------+---------------------------+---------------------------+
   | Started from                                   | DIOSSA-1 MobilePose       | blest-harl                |
   +------------------------------------------------+---------------------------+---------------------------+
   | Input                                          | 224 × 224                 | 224 × 224                 |
   |                                                | (selected over 448 × 448) |                           |
   +------------------------------------------------+---------------------------+---------------------------+
   | Keypoints                                      | 20                        | 20                        |
   +------------------------------------------------+---------------------------+---------------------------+
   | L2 pixel error                                 | 3.799                     | —                         |
   | (reference frame not stated in the prose)      |                           |                           |
   +------------------------------------------------+---------------------------+---------------------------+
   | L1 on the camera image [2590 × 1942], px       | 4.783                     | 5.303                     |
   +------------------------------------------------+---------------------------+---------------------------+
   | L1 on the network input, px                    | not stated                | 1.564                     |
   +------------------------------------------------+---------------------------+---------------------------+
   | L2 on the camera image [2590 × 1942], px       | not stated                | 4.216                     |
   +------------------------------------------------+---------------------------+---------------------------+
   | L2 on the network input, px                    | not stated                | 1.238                     |
   +------------------------------------------------+---------------------------+---------------------------+

The accepted quantization loss is 0.520 px L1 on the camera image
(4.783 to 5.303). Two other QAT learning rates were worse
(``smoky-quey`` 6.848 px, ``swish-mows`` 5.797 px), so ``bijou-rasp``
is the CCN1 selection.

Compiled models on the Xilinx DPU
----------------------------------

AD-1 confirms that both networks compile with Vitis-AI and run on the
DPU. The embedded pairs used for the trajectory evaluation are:

- 20 m to 100 m: ``edge-bijou-rasp`` with ``edge-quare-delf``.
- 20 m to 300 m: ``edge-bijou-rasp`` with ``edge-livid-wire``.

AD-1 does not publish a separate IoU or keypoint error for those
compiled binaries. The task metrics above are the PyTorch QAT
evaluation. The edge result in AD-1 is the end-to-end pose, given as
image tables rather than text, plus a qualitative check that the DPU
boxes and keypoints lie close to the ground truth on a sample frame.

What the prose does state:

- On the float pair, relative position error stays under 2% at
  1-sigma on trajectories 1, 2, 3, 4 and 5, with good behaviour also
  on trajectories 4 and 7.
- On the QAT pair running on the DPU, relative position error stays
  under 2% at 1-sigma on trajectories 2 and 3.
- Trajectory 1 shows a rotation error near 180 degrees. AD-1 attributes
  that to symmetry of the chosen keypoints, not to quantization. The
  float models already place the position correctly and fail the
  rotation the same way.

Those pose tables are the right place to judge the DPU, and they are
not transcribed in AD-1. Reading the figures back into this note is
the remaining Xilinx-side action.

PolarFire measurements
======================

The same QAT checkpoints are compiled to VNNX for the VectorBlox core.
Post-processing that the core does not absorb runs as ONNX on the
PolarFire SoC RISC-V cores. No accuracy or latency figure from that
path is available yet.

.. table:: Edge comparison. PolarFire column to be filled from the current runs.
   :width: 100%
   :widths: auto

   +----------------------------------------------+---------------------------+---------------------------+
   |                                              | **Xilinx DPU**            | **PolarFire VectorBlox**  |
   +----------------------------------------------+---------------------------+---------------------------+
   | FCOS checkpoint                              | quare-delf                | quare-delf, epoch 19      |
   +----------------------------------------------+---------------------------+---------------------------+
   | FCOS input                                   | 384 × 288                 | 384 × 288                 |
   +----------------------------------------------+---------------------------+---------------------------+
   | FCOS IoU on the shared sample                | not tabulated in AD-1     | TBD                       |
   +----------------------------------------------+---------------------------+---------------------------+
   | FCOS mAP at IoU > 0.75                       | not tabulated in AD-1     | TBD                       |
   +----------------------------------------------+---------------------------+---------------------------+
   | FCOS accelerator latency                     | not reported in AD-1      | TBD                       |
   +----------------------------------------------+---------------------------+---------------------------+
   | FCOS end-to-end latency, including the       | not reported in AD-1      | TBD                       |
   | CPU post-processing                          |                           |                           |
   +----------------------------------------------+---------------------------+---------------------------+
   | MobilePose checkpoint                        | bijou-rasp                | bijou-rasp, epoch 17      |
   +----------------------------------------------+---------------------------+---------------------------+
   | MobilePose input                             | 224 × 224, 20 keypoints   | 224 × 224, 20 keypoints   |
   +----------------------------------------------+---------------------------+---------------------------+
   | MobilePose L1 on the camera image, px        | not tabulated in AD-1     | TBD                       |
   +----------------------------------------------+---------------------------+---------------------------+
   | MobilePose L2 on the network input, px       | not tabulated in AD-1     | TBD                       |
   +----------------------------------------------+---------------------------+---------------------------+
   | MobilePose accelerator latency               | not reported in AD-1      | TBD                       |
   +----------------------------------------------+---------------------------+---------------------------+
   | MobilePose end-to-end latency                | not reported in AD-1      | TBD                       |
   +----------------------------------------------+---------------------------+---------------------------+

The sample used for the PolarFire runs is the CCN1 Inmarsat-5 test
sample already staged for these two input sizes. The Xilinx edge
numbers, once read out of the AD-1 pose tables or re-measured, have to
be computed on that same sample. A comparison against the full CCN1
validation set would mix two test distributions.

How the result will be judged
=============================

The switch is supported if both of the following hold on the shared
sample.

Task accuracy
-------------

The PolarFire edge network stays at or above the CCN1 QAT validation
score, within the quantization loss that ESA already accepted when the
DPU path was approved:

- FCOS at 384 × 288: IoU at least 0.858, or a documented shortfall
  smaller than the 0.067 already lost between ``licit-weal`` and
  ``quare-delf``.
- MobilePose: L1 on the 2590 × 1942 image at most 5.303 px, or a
  documented excess smaller than the 0.520 px already lost between
  ``blest-harl`` and ``bijou-rasp``.

A gap inside that band is the same kind of loss as the quantization
step, not a new loss introduced by PolarFire. A gap larger than that
band has to be explained from the compile log (unsupported operators
moved to the RISC-V head, scale and zero-point handling, or a
difference in the test sample) before any claim is made.

These thresholds are the CCN1 validation-set scores. The PolarFire
campaign runs on a smaller sample, so the numerical test is the
difference between Xilinx edge and PolarFire edge on that sample, not
a direct comparison of the sample score with 0.858 or 5.303.

Latency
-------

AD-1 gives no DPU latency, so a numerical latency requirement cannot
be taken from the previous file. The RPO chain is detection, then
keypoint regression, then the pose solver. The budget that matters is
the sum of the two accelerator times and the two CPU post-processing
times, at the image rate the payload is required to close. That rate
is not stated in AD-1 and has to be taken from the payload
specification when the PolarFire times are in.

Open items
==========

#. Transcribe the AD-1 edge pose tables (the third row of each table,
   which is the DPU evaluation of ``edge-bijou-rasp`` with
   ``edge-quare-delf`` and with ``edge-livid-wire``) so the Xilinx
   edge column is numeric.
#. Confirm the reference frame of the ``blest-harl`` L2 figure of
   3.799 px from the float experiment figure in AD-1.
#. Fill the PolarFire column: IoU, mAP, L1, L2, accelerator time, and
   end-to-end time, on the staged CCN1 sample, for ``quare-delf`` and
   ``bijou-rasp``.
#. Re-measure the Xilinx DPU on that same sample, or show that the
   AD-1 edge tables already cover it.
