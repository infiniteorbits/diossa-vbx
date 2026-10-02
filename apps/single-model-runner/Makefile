# =========================
# On-target Makefile
# =========================

# Compilers (override in env if needed)
CC ?= gcc
CXX ?= g++

# ONNX Runtime install produced for this board (headers + libonnxruntime.so).
ORT_DIR ?= ../../ort/output

# Select target hardware
kit = videokit
ifeq ($(kit),discovery)
	C_FLAGS += -DUSE_INTERRUPTS=0 -DMSS_DDR
else ifeq ($(kit),icicle)
	C_FLAGS += -DUSE_INTERRUPTS=0 -DMSS_DDR -DPDMA
else
	C_FLAGS += -DUSE_INTERRUPTS=1 -DPDMA
endif

all: run-model

# pdma_helpers only needed when PDMA is defined (non-discovery kits)
ifneq ($(kit),discovery)
C_SRCS = pdma/pdma_helpers.c
else
C_SRCS =
endif
C_SRCS += ../postprocess/image.c
C_SRCS += ../../drivers/vectorblox/vbx_cnn_api.c ../../drivers/vectorblox/vbx_cnn_model.c
DEBUG_SRC = ../../drivers/vectorblox/vbx_cnn_api_debug.c
ifneq ($(wildcard $(DEBUG_SRC)),)
  C_SRCS += $(DEBUG_SRC)
  C_FLAGS += -DDEBUG_CYCLE
endif

CXX_SRCS = run-model.cpp
C_OBJS = $(addsuffix .o,$(addprefix obj/,$(abspath $(C_SRCS))))
CXX_OBJS = $(addsuffix .o,$(addprefix obj/,$(abspath $(CXX_SRCS))))

C_FLAGS += -Wall -O3 -I../../drivers/vectorblox/ -Ipdma/ -MD -DVBX_SOC_DRIVER
CXX_FLAGS = $(C_FLAGS) -std=c++17 -I$(ORT_DIR)/include

$(C_OBJS): obj/%.o: %
	mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -c $< -o $@

$(CXX_OBJS): obj/%.o: %
	mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

run-model: $(CXX_OBJS) $(C_OBJS)
	$(CXX) -o $@ $^ -ljpeg -L$(ORT_DIR) -lonnxruntime \
		-Wl,-rpath,'$$ORIGIN' \
		-Wl,-rpath,'$$ORIGIN/../../ort/output'

# Copy the runtime next to the binary so $$ORIGIN finds libonnxruntime.so.1 on the board.
.PHONY: stage
stage: run-model
	cp -a $(ORT_DIR)/libonnxruntime.so.1 .

.PHONY: overlay
overlay:
	cd dts && dtc -@ -O dtb -o mpfs_vbx.dtbo mpfs_vbx.dtso -Wno-reg_format -Wno-avoid_default_addr_size && \
	bash overlay.sh mpfs_vbx.dtbo

.PHONY: clean
clean:
	rm -rf run-model libonnxruntime.so.1 */*.dtbo *.o */*.o *.a obj
