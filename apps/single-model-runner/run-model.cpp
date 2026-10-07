#include "vbx_cnn_api.h"
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <vector>

#ifdef PDMA
#include "pdma/pdma_helpers.h"
#endif

extern "C" int read_JPEG_file(char* filename, int* width, int* height,
		unsigned char** image, const int grayscale);
extern "C" void resize_image(uint8_t* image_in, int in_w, int in_h,
		uint8_t* image_out, int out_w, int out_h);

#ifndef USE_INTERRUPTS
#define USE_INTERRUPTS 1
#endif

#ifndef NUM_LOOPS
#define NUM_LOOPS 1
#endif

#ifndef DEBUG_CYCLE
#define DEBUG_CYCLE 0
#endif

static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn, void* virt) {
	return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}

#if USE_INTERRUPTS
static void enable_interrupt(vbx_cnn_t* vbx_cnn) {
	uint32_t reenable = 1;
	ssize_t writeSize = write(vbx_cnn->fd, &reenable, sizeof(uint32_t));
	if (writeSize < 0) {
		close(vbx_cnn->fd);
	}
}
#endif

#ifdef PDMA
static int8_t* pdma_mmap_t;

static uint64_t pdma_mmap(size_t total_size) {
	char cdev[256] = "/dev/udmabuf-ddr-c0";
	uint64_t ddrc_phyadr = get_phy_addr(cdev);
	int32_t fdc = open(cdev, O_RDWR);
	off_t oft = 0;
	pdma_mmap_t = (int8_t*)mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fdc, oft);
	assert(pdma_mmap_t != MAP_FAILED);
	return ddrc_phyadr + oft;
}

static int32_t pdma_ch_transfer(uint64_t output_data_phys, void* source_buffer, int offset, int size, vbx_cnn_t* vbx_cnn, int32_t channel) {
	uint64_t srcbuf = 0x3000000000ULL + (uint64_t)(uintptr_t)virt_to_phys(vbx_cnn, source_buffer);
	return pdma_ch_cpy(output_data_phys + offset, srcbuf, size, channel);
}
#endif

static void* read_image(const char* filename, int channels, int height, int width, int use_bgr) {
	unsigned char* image;
	int h, w;
	read_JPEG_file((char*)filename, &w, &h, &image, channels == 1);
	unsigned char* planer_img = (unsigned char*)malloc(w * h * channels);
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			for (int ch = 0; ch < channels; ch++) {
				if (use_bgr) {
					planer_img[ch * w * h + r * w + c] = image[(r * w + c) * channels + ((channels - 1) - ch)];
				} else {
					planer_img[ch * w * h + r * w + c] = image[(r * w + c) * channels + ch];
				}
			}
		}
	}
	free(image);

	unsigned char* resized_planar_img = (unsigned char*)malloc(width * height * channels);
	for (int ch = 0; ch < channels; ch++) {
		resize_image((uint8_t*)planer_img + ch * w * h, w, h,
				(uint8_t*)resized_planar_img + ch * width * height, width, height);
	}
	free(planer_img);
	return resized_planar_img;
}

static ucomp_model_t* read_ucompmodel_file(vbx_cnn_t* vbx_cnn, const char* filename) {
	ucomp_model_t* model_info = (ucomp_model_t*)malloc(sizeof(ucomp_model_t));
	FILE* model_file = fopen(filename, "r");
	if (model_file == NULL) {
		return NULL;
	}
	fseek(model_file, 0, SEEK_END);
	int file_size = ftell(model_file);
	fseek(model_file, 0, SEEK_SET);
	model_t* model = (model_t*)malloc(file_size);
	int size_read = fread(model, 1, file_size, model_file);
	if (size_read != file_size) {
		fprintf(stderr, "Error reading full model file %s\n", filename);
		return NULL;
	}
	fclose(model_file);

	uint32_t header_size = *((uint32_t*)model);
	uint32_t mxp_model_size = *((uint32_t*)model + 1);
	uint32_t tsnp_model_size = *((uint32_t*)model + 2);

	model_info->header_info = (uint8_t*)vbx_allocate_dma_buffer(vbx_cnn, header_size, 0);
	if (model_info->header_info) {
		memcpy(model_info->header_info, model, header_size);
	} else {
		return 0;
	}

	uint32_t model_allocate_size = model_get_allocate_bytes((model_t*)((char*)model + header_size));
	model_info->mxp_model = (model_t*)vbx_allocate_dma_buffer(vbx_cnn, model_allocate_size, 0);
	if (model_info->mxp_model) {
		memcpy(model_info->mxp_model, (model_t*)((char*)model + header_size), mxp_model_size);
	} else {
		return NULL;
	}

	model_info->tsnp_model = (model_t*)vbx_allocate_dma_buffer(vbx_cnn, tsnp_model_size, 12);
	if (model_info->tsnp_model) {
		memcpy(model_info->tsnp_model, (model_t*)((char*)model + header_size + mxp_model_size), tsnp_model_size);
	} else {
		return NULL;
	}

	free(model);
	return model_info;
}

static model_t* read_model_file(vbx_cnn_t* vbx_cnn, const char* filename) {
	FILE* model_file = fopen(filename, "r");
	if (model_file == NULL) {
		return NULL;
	}
	fseek(model_file, 0, SEEK_END);
	int file_size = ftell(model_file);
	fseek(model_file, 0, SEEK_SET);
	model_t* model = (model_t*)malloc(file_size);
	printf("Reading model\n");
	int size_read = fread(model, 1, file_size, model_file);
	printf("Done\n");
	if (size_read != file_size) {
		fprintf(stderr, "Error reading full model file %s\n", filename);
		return NULL;
	}
	int model_data_size = model_get_data_bytes(model);
	if (model_data_size != file_size) {
		fprintf(stderr, "Error model file is not correct size%s\n", filename);
		return NULL;
	}
	int model_allocate_size = model_get_allocate_bytes(model);
	model = (model_t*)realloc(model, model_allocate_size);
	model_t* dma_model = (model_t*)vbx_allocate_dma_buffer(vbx_cnn, model_allocate_size, 0);
	if (dma_model) {
		memcpy(dma_model, model, model_data_size);
	}
	free(model);
	return dma_model;
}

static int gettimediff_us(struct timeval start, struct timeval end) {
	int sec = end.tv_sec - start.tv_sec;
	int usec = end.tv_usec - start.tv_usec;
	return sec * 1000000 + usec;
}

static uint32_t fletcher32(const uint16_t* data, size_t len) {
	uint32_t c0, c1;
	unsigned int i;

	for (c0 = c1 = 0; len >= 360; len -= 360) {
		for (i = 0; i < 360; ++i) {
			c0 = c0 + *data++;
			c1 = c1 + c0;
		}
		c0 = c0 % 65535;
		c1 = c1 % 65535;
	}
	for (i = 0; i < len; ++i) {
		c0 = c0 + *data++;
		c1 = c1 + c0;
	}
	c0 = c0 % 65535;
	c1 = c1 % 65535;
	return (c1 << 16 | c0);
}

static int calc_type_bytes(vbx_cnn_calc_type_e type) {
	switch (type) {
	case VBX_CNN_CALC_TYPE_INT16:
		return 2;
	case VBX_CNN_CALC_TYPE_INT32:
		return 4;
	default:
		return 1;
	}
}

static const char* vbx_type_name(vbx_cnn_calc_type_e type) {
	switch (type) {
	case VBX_CNN_CALC_TYPE_UINT8:
		return "uint8";
	case VBX_CNN_CALC_TYPE_INT8:
		return "int8";
	case VBX_CNN_CALC_TYPE_INT16:
		return "int16";
	case VBX_CNN_CALC_TYPE_INT32:
		return "int32";
	default:
		return "unknown";
	}
}

static const char* onnx_type_name(ONNXTensorElementDataType type) {
	switch (type) {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return "float32";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		return "uint8";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		return "int8";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		return "int16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		return "int32";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		return "int64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		return "float64";
	default:
		return "other";
	}
}

static ONNXTensorElementDataType vbx_to_onnx_type(vbx_cnn_calc_type_e type) {
	switch (type) {
	case VBX_CNN_CALC_TYPE_UINT8:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
	case VBX_CNN_CALC_TYPE_INT8:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
	case VBX_CNN_CALC_TYPE_INT16:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
	case VBX_CNN_CALC_TYPE_INT32:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	default:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	}
}

static std::string shape_string(const std::vector<int64_t>& shape) {
	std::ostringstream out;
	out << "[";
	for (size_t i = 0; i < shape.size(); ++i) {
		if (i) {
			out << ", ";
		}
		out << shape[i];
	}
	out << "]";
	return out.str();
}

static std::vector<int64_t> vbx_output_shape(const model_t* model, int index) {
	size_t rank = model_get_output_dims(model, index);
	int* raw = model_get_output_shape(model, index);
	std::vector<int64_t> shape(rank);
	for (size_t i = 0; i < rank; ++i) {
		shape[i] = raw[i];
	}
	return shape;
}

// Same rank, and every fixed ONNX dim equals the VBX dim. A negative ONNX dim
// matches any VBX size.
static bool static_dims_match(const std::vector<int64_t>& onnx_shape,
		const std::vector<int64_t>& vbx_shape) {
	if (onnx_shape.size() != vbx_shape.size()) {
		return false;
	}
	for (size_t i = 0; i < onnx_shape.size(); ++i) {
		if (onnx_shape[i] >= 0 && onnx_shape[i] != vbx_shape[i]) {
			return false;
		}
	}
	return true;
}

// True when resolve_shape would accept this pair without throwing.
static bool can_resolve_shape(const std::vector<int64_t>& onnx_shape,
		const std::vector<int64_t>& vbx_shape, size_t vbx_len) {
	if (static_dims_match(onnx_shape, vbx_shape)) {
		return true;
	}
	if (onnx_shape.size() == vbx_shape.size()) {
		return false;
	}
	int dynamic = 0;
	int64_t known = 1;
	for (int64_t dim : onnx_shape) {
		if (dim < 0) {
			dynamic++;
		} else {
			known *= dim;
		}
	}
	if (dynamic == 0 && known == (int64_t)vbx_len) {
		return true;
	}
	return dynamic == 1 && known > 0 && (vbx_len % (size_t)known) == 0;
}

// VNNX outputs follow compiler execution order. The CPU graph follows the
// order the backbone returned them. FCOS returns five box maps, then five
// class maps, then five centerness maps, while the VNNX file emits each
// pyramid level as it is produced (P6, P7, P5, P4, P3 on these models).
// Index pairing then compares P3 [1, 4, 36, 48] with P6 [1, 4, 5, 6].
//
// Pairing walks ONNX inputs from 0 upward and gives each the lowest-index
// unused VBX output of the same shape. A shape that occurs more than once
// keeps that index order on both sides: the first such ONNX input gets the
// first such VBX output, the second gets the second, and so on.
static std::vector<size_t> pair_vbx_outputs(const std::vector<std::string>& input_names,
		const std::vector<std::vector<int64_t>>& onnx_shapes,
		const std::vector<std::vector<int64_t>>& vbx_shapes,
		const std::vector<size_t>& vbx_lengths) {
	size_t count = onnx_shapes.size();
	std::vector<size_t> identity(count);
	bool index_order = true;
	for (size_t i = 0; i < count; ++i) {
		identity[i] = i;
		if (!can_resolve_shape(onnx_shapes[i], vbx_shapes[i], vbx_lengths[i])) {
			index_order = false;
		}
	}
	if (index_order) {
		return identity;
	}

	std::vector<size_t> order(count);
	std::vector<char> used(count, 0);
	size_t shared = 0;
	for (size_t i = 0; i < count; ++i) {
		size_t matches = 0;
		size_t chosen = count;
		for (size_t j = 0; j < count; ++j) {
			if (used[j] || !static_dims_match(onnx_shapes[i], vbx_shapes[j])) {
				continue;
			}
			matches++;
			if (chosen == count) {
				chosen = j;
			}
		}
		if (chosen == count) {
			std::ostringstream out;
			out << "VNNX output order does not match the ONNX inputs, and ONNX input "
				<< i << " " << input_names[i] << " " << shape_string(onnx_shapes[i])
				<< " has no unused VBX output of that shape.\n";
			for (size_t n = 0; n < count; ++n) {
				out << "ONNX input " << n << " " << input_names[n] << " " << shape_string(onnx_shapes[n]) << "\n";
			}
			for (size_t n = 0; n < count; ++n) {
				out << "VBX output " << n << " " << shape_string(vbx_shapes[n]) << "\n";
			}
			throw std::runtime_error(out.str());
		}
		if (matches > 1) {
			shared++;
		}
		used[chosen] = 1;
		order[i] = chosen;
	}

	printf("VNNX outputs are not in ONNX input order. Pairing the %zu tensors by shape.\n", count);
	if (shared > 0) {
		printf("  %zu ONNX inputs share a shape with another input. "
				"Those are paired in index order: earliest ONNX input with earliest VBX output.\n",
				shared);
	}
	return order;
}

// Fill dynamic ONNX dims from the VBX tensor. A rank change is accepted when
// the element count matches, which is a reshape of the same contiguous buffer.
static std::vector<int64_t> resolve_shape(const std::vector<int64_t>& onnx_shape,
		const std::vector<int64_t>& vbx_shape, size_t vbx_len, bool* reshaped) {
	*reshaped = false;
	if (onnx_shape.size() == vbx_shape.size()) {
		std::vector<int64_t> resolved = onnx_shape;
		for (size_t i = 0; i < resolved.size(); ++i) {
			if (resolved[i] < 0) {
				resolved[i] = vbx_shape[i];
			} else if (resolved[i] != vbx_shape[i]) {
				throw std::runtime_error("ONNX dimension " + std::to_string(i) +
						" is " + std::to_string(resolved[i]) +
						" but the VBX output is " + std::to_string(vbx_shape[i]));
			}
		}
		return resolved;
	}

	int dynamic = 0;
	int64_t known = 1;
	for (int64_t dim : onnx_shape) {
		if (dim < 0) {
			dynamic++;
		} else {
			known *= dim;
		}
	}
	if (dynamic == 0 && known == (int64_t)vbx_len) {
		*reshaped = true;
		return onnx_shape;
	}
	if (dynamic == 1 && known > 0 && (vbx_len % (size_t)known) == 0) {
		std::vector<int64_t> resolved = onnx_shape;
		for (int64_t& dim : resolved) {
			if (dim < 0) {
				dim = (int64_t)(vbx_len / (size_t)known);
			}
		}
		*reshaped = true;
		return resolved;
	}
	throw std::runtime_error("cannot match ONNX shape " + shape_string(onnx_shape) +
			" to VBX shape " + shape_string(vbx_shape));
}

static int32_t quantized_at(const void* src, size_t index, vbx_cnn_calc_type_e type) {
	switch (type) {
	case VBX_CNN_CALC_TYPE_UINT8:
		return ((const uint8_t*)src)[index];
	case VBX_CNN_CALC_TYPE_INT8:
		return ((const int8_t*)src)[index];
	case VBX_CNN_CALC_TYPE_INT16:
		return ((const int16_t*)src)[index];
	case VBX_CNN_CALC_TYPE_INT32:
		return ((const int32_t*)src)[index];
	default:
		return ((const int8_t*)src)[index];
	}
}

static void dequantize(float* dst, const void* src, size_t count, vbx_cnn_calc_type_e type, float scale, int zero_point) {
	for (size_t i = 0; i < count; ++i) {
		dst[i] = (float)(quantized_at(src, i, type) - zero_point) * scale;
	}
}

template <typename T>
static void print_preview(const char* label, const T* data, size_t count) {
	size_t show = count < 8 ? count : 8;
	printf("  %s %zu:", label, show);
	for (size_t i = 0; i < show; ++i) {
		printf(" %.6g", (double)data[i]);
	}
	printf("\n");
}

static void print_float_output(const std::string& name, const std::vector<int64_t>& shape, const float* data, size_t count) {
	float min_v = 0.f;
	float max_v = 0.f;
	size_t argmax = 0;
	if (count > 0) {
		min_v = max_v = data[0];
		for (size_t i = 1; i < count; ++i) {
			if (data[i] < min_v) {
				min_v = data[i];
			}
			if (data[i] > max_v) {
				max_v = data[i];
				argmax = i;
			}
		}
	}
	printf("output %s float32 %s  min=%.6g max=%.6g argmax=%zu (%.6g)\n",
			name.c_str(), shape_string(shape).c_str(), min_v, max_v, argmax, count ? data[argmax] : 0.f);
	print_preview("first", data, count);

	if (count > 1 && count <= 4096) {
		int k = count < 5 ? (int)count : 5;
		std::vector<size_t> order(count);
		for (size_t i = 0; i < count; ++i) {
			order[i] = i;
		}
		std::partial_sort(order.begin(), order.begin() + k, order.end(),
				[&](size_t a, size_t b) { return data[a] > data[b]; });
		printf("  top %d:", k);
		for (int i = 0; i < k; ++i) {
			printf(" [%zu]=%.6g", order[i], data[order[i]]);
		}
		printf("\n");
	}
}

static std::string json_path_for_image(const std::string& image_path) {
	std::string path = image_path;
	size_t slash = path.find_last_of('/');
	size_t dot = path.find_last_of('.');
	if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
		path.resize(dot);
	}
	path += ".json";
	return path;
}

static void write_json_string(FILE* fp, const std::string& text) {
	fputc('"', fp);
	for (unsigned char c : text) {
		switch (c) {
		case '"':
			fputs("\\\"", fp);
			break;
		case '\\':
			fputs("\\\\", fp);
			break;
		case '\n':
			fputs("\\n", fp);
			break;
		case '\r':
			fputs("\\r", fp);
			break;
		case '\t':
			fputs("\\t", fp);
			break;
		default:
			if (c < 0x20) {
				fprintf(fp, "\\u%04x", c);
			} else {
				fputc(c, fp);
			}
		}
	}
	fputc('"', fp);
}

template <typename T>
static void write_json_numbers(FILE* fp, const T* data, size_t count, bool as_float) {
	fputc('[', fp);
	for (size_t i = 0; i < count; ++i) {
		if (i) {
			fputc(',', fp);
		}
		if (as_float) {
			double value = (double)data[i];
			if (std::isfinite(value)) {
				fprintf(fp, "%.9g", value);
			} else {
				fputs("null", fp);
			}
		} else {
			fprintf(fp, "%lld", (long long)data[i]);
		}
	}
	fputc(']', fp);
}

static void write_onnx_outputs_json(const char* path, std::vector<Ort::Value>& outputs, const std::vector<std::string>& names) {
	FILE* fp = fopen(path, "w");
	if (!fp) {
		throw std::runtime_error(std::string("Unable to write ") + path);
	}
	fputs("{\n\"outputs\":[\n", fp);
	bool first = true;
	for (size_t i = 0; i < outputs.size(); ++i) {
		if (!outputs[i].IsTensor()) {
			continue;
		}
		if (!first) {
			fputs(",\n", fp);
		}
		first = false;
		auto info = outputs[i].GetTensorTypeAndShapeInfo();
		auto shape = info.GetShape();
		size_t count = info.GetElementCount();
		ONNXTensorElementDataType type = info.GetElementType();
		fputs("{", fp);
		fputs("\"name\":", fp);
		write_json_string(fp, names[i]);
		fprintf(fp, ",\"dtype\":\"%s\",\"shape\":[", onnx_type_name(type));
		for (size_t d = 0; d < shape.size(); ++d) {
			if (d) {
				fputc(',', fp);
			}
			fprintf(fp, "%lld", (long long)shape[d]);
		}
		fputs("],\"data\":", fp);
		if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
			write_json_numbers(fp, outputs[i].GetTensorData<float>(), count, true);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
			write_json_numbers(fp, outputs[i].GetTensorData<double>(), count, true);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
			write_json_numbers(fp, outputs[i].GetTensorData<int64_t>(), count, false);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
			write_json_numbers(fp, outputs[i].GetTensorData<int32_t>(), count, false);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
			write_json_numbers(fp, outputs[i].GetTensorData<int16_t>(), count, false);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
			write_json_numbers(fp, outputs[i].GetTensorData<int8_t>(), count, false);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
			write_json_numbers(fp, outputs[i].GetTensorData<uint8_t>(), count, false);
		} else {
			fputs("null", fp);
		}
		fputc('}', fp);
	}
	fputs("\n]}\n", fp);
	fclose(fp);
}

static void print_onnx_outputs(std::vector<Ort::Value>& outputs, const std::vector<std::string>& names) {
	for (size_t i = 0; i < outputs.size(); ++i) {
		if (!outputs[i].IsTensor()) {
			printf("output %s is not a tensor\n", names[i].c_str());
			continue;
		}
		auto info = outputs[i].GetTensorTypeAndShapeInfo();
		auto shape = info.GetShape();
		size_t count = info.GetElementCount();
		ONNXTensorElementDataType type = info.GetElementType();
		if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
			print_float_output(names[i], shape, outputs[i].GetTensorData<float>(), count);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
			printf("output %s int64 %s\n", names[i].c_str(), shape_string(shape).c_str());
			print_preview("first", outputs[i].GetTensorData<int64_t>(), count);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
			printf("output %s int32 %s\n", names[i].c_str(), shape_string(shape).c_str());
			print_preview("first", outputs[i].GetTensorData<int32_t>(), count);
		} else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
			printf("output %s float64 %s\n", names[i].c_str(), shape_string(shape).c_str());
			print_preview("first", outputs[i].GetTensorData<double>(), count);
		} else {
			printf("output %s %s %s (%zu elements)\n", names[i].c_str(), onnx_type_name(type),
					shape_string(shape).c_str(), count);
		}
	}
}

static void run_onnx_head(const char* onnx_path, const char* image_path, const model_t* model, const std::vector<uint8_t*>& cpu_outputs) {
	int threads = 1;
	if (const char* env = getenv("ORT_NUM_THREADS")) {
		threads = atoi(env);
		if (threads < 1) {
			threads = 1;
		}
	}

	printf("ONNX Runtime %s\n", Ort::GetVersionString().c_str());
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vbx-onnx");
	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads(threads);
	opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
	Ort::Session session(env, onnx_path, opts);

	size_t num_vbx = model_get_num_outputs(model);
	size_t num_inputs = session.GetInputCount();
	if (num_inputs != num_vbx) {
		throw std::runtime_error("ONNX model has " + std::to_string(num_inputs) +
				" inputs but the VNNX model has " + std::to_string(num_vbx) +
				" outputs. Export the CPU graph so each input is one VBX output, in the same order.");
	}

	std::vector<std::string> input_names = session.GetInputNames();
	std::vector<std::string> output_names = session.GetOutputNames();
	std::vector<const char*> input_name_ptrs;
	std::vector<const char*> output_name_ptrs;
	input_name_ptrs.reserve(input_names.size());
	output_name_ptrs.reserve(output_names.size());
	for (const std::string& name : input_names) {
		input_name_ptrs.push_back(name.c_str());
	}
	for (const std::string& name : output_names) {
		output_name_ptrs.push_back(name.c_str());
	}

	Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<std::vector<float>> float_inputs(num_inputs);
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.reserve(num_inputs);

	std::vector<std::vector<int64_t>> onnx_shapes(num_inputs);
	std::vector<ONNXTensorElementDataType> onnx_types(num_inputs);
	for (size_t i = 0; i < num_inputs; ++i) {
		auto type_info = session.GetInputTypeInfo(i);
		if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
			throw std::runtime_error("ONNX input " + input_names[i] + " is not a tensor");
		}
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		onnx_types[i] = tensor_info.GetElementType();
		onnx_shapes[i] = tensor_info.GetShape();
	}
	std::vector<std::vector<int64_t>> vbx_shapes(num_inputs);
	std::vector<size_t> vbx_lengths(num_inputs);
	for (size_t i = 0; i < num_inputs; ++i) {
		vbx_shapes[i] = vbx_output_shape(model, (int)i);
		vbx_lengths[i] = model_get_output_length(model, (int)i);
	}
	std::vector<size_t> vbx_for_onnx = pair_vbx_outputs(input_names, onnx_shapes, vbx_shapes, vbx_lengths);

	struct timeval prep_start, prep_end;
	gettimeofday(&prep_start, NULL);
	for (size_t i = 0; i < num_inputs; ++i) {
		size_t vbx_index = vbx_for_onnx[i];
		ONNXTensorElementDataType onnx_type = onnx_types[i];
		std::vector<int64_t> onnx_shape = onnx_shapes[i];

		vbx_cnn_calc_type_e vbx_type = model_get_output_datatype(model, (int)vbx_index);
		std::vector<int64_t> vbx_shape = vbx_shapes[vbx_index];
		size_t length = vbx_lengths[vbx_index];
		float scale = model_get_output_scale_value(model, (int)vbx_index);
		int zero_point = model_get_output_zeropoint(model, (int)vbx_index);
		bool reshaped = false;
		std::vector<int64_t> shape = resolve_shape(onnx_shape, vbx_shape, length, &reshaped);
		int64_t resolved_count = 1;
		for (int64_t dim : shape) {
			resolved_count *= dim;
		}
		if (resolved_count != (int64_t)length) {
			throw std::runtime_error("ONNX input " + input_names[i] + " resolves to " +
					std::to_string(resolved_count) + " elements but VBX output " + std::to_string(vbx_index) +
					" has " + std::to_string(length));
		}

		printf("VBX output %zu: %s %s scale=%g zero=%d\n", vbx_index, vbx_type_name(vbx_type),
				shape_string(vbx_shape).c_str(), scale, zero_point);

		if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
			float_inputs[i].resize(length);
			dequantize(float_inputs[i].data(), cpu_outputs[vbx_index], length, vbx_type, scale, zero_point);
			ort_inputs.push_back(Ort::Value::CreateTensor<float>(
					memory, float_inputs[i].data(), length, shape.data(), shape.size()));
			printf("  -> ONNX input %s float32 %s (dequantized)\n", input_names[i].c_str(), shape_string(shape).c_str());
		} else if (onnx_type == vbx_to_onnx_type(vbx_type)) {
			size_t nbytes = length * (size_t)calc_type_bytes(vbx_type);
			ort_inputs.push_back(Ort::Value::CreateTensor(
					memory, cpu_outputs[vbx_index], nbytes, shape.data(), shape.size(), onnx_type));
			printf("  -> ONNX input %s %s %s (raw accelerator values)\n", input_names[i].c_str(),
					onnx_type_name(onnx_type), shape_string(shape).c_str());
		} else {
			throw std::runtime_error("ONNX input " + input_names[i] + " is " + onnx_type_name(onnx_type) +
					" but VBX output " + std::to_string(vbx_index) + " is " + vbx_type_name(vbx_type) +
					". Use float32 to receive dequantized values, or the same integer type to receive raw values.");
		}
		if (reshaped) {
			printf("  note: bytes stay in VBX order and are viewed as %s. Add a Transpose to the ONNX graph if the head expects a different layout.\n",
					shape_string(shape).c_str());
		}
	}
	gettimeofday(&prep_end, NULL);

	struct timeval run_start, run_end;
	gettimeofday(&run_start, NULL);
	Ort::RunOptions run_options;
	std::vector<Ort::Value> ort_outputs = session.Run(
			run_options,
			input_name_ptrs.data(), ort_inputs.data(), ort_inputs.size(),
			output_name_ptrs.data(), output_name_ptrs.size());
	gettimeofday(&run_end, NULL);

	printf("dequantize took %3.4f ms, ONNX Runtime took %3.4f ms\n",
			gettimediff_us(prep_start, prep_end) / 1000.0,
			gettimediff_us(run_start, run_end) / 1000.0);
	print_onnx_outputs(ort_outputs, output_names);
	const char* write_out = getenv("WRITE_OUT");
	if (write_out && std::strcmp(write_out, "1") == 0) {
		std::string out_path = json_path_for_image(image_path);
		write_onnx_outputs_json(out_path.c_str(), ort_outputs, output_names);
		printf("Wrote %s\n", out_path.c_str());
	}
}

int main(int argc, char** argv) {
	if (argc < 4) {
		fprintf(stderr,
				"Usage: %s MODEL.vnnx IMAGE.jpg HEAD.onnx\n"
				"  IMAGE.jpg may be TEST_DATA to use the vectors stored in the VNNX file.\n"
				"  HEAD.onnx runs on the CPU. Its inputs are the VNNX outputs.\n"
				"  Inputs are paired by index when the shapes match, otherwise by shape.\n"
				"  A repeated shape is paired in index order on both sides.\n"
				"  float32 inputs are dequantized as (q - zero_point) * scale.\n"
				"  Matching integer inputs receive the raw accelerator values.\n",
				argv[0]);
		return 1;
	}

	vbx_cnn_t* vbx_cnn = vbx_cnn_init(NULL);
	if (!vbx_cnn) {
		fprintf(stderr, "Unable to initialize vbx_cnn. Exiting\n");
		return 1;
	}

	model_t* model = NULL;
	model_t* tsnp_model = NULL;
	uint8_t* model_header = NULL;
	uint32_t input_offset = 0;
	if (vbx_cnn->comp_config == 2) {
		ucomp_model_t* ucomp_model = read_ucompmodel_file(vbx_cnn, argv[1]);
		if (!ucomp_model) {
			fprintf(stderr, "Unable to correctly read %s. Exiting\n", argv[1]);
			return 1;
		}
		model = ucomp_model->mxp_model;
		tsnp_model = ucomp_model->tsnp_model;
		model_header = ucomp_model->header_info;
		input_offset = *((uint32_t*)model_header + 6);
	} else {
		model = read_model_file(vbx_cnn, argv[1]);
	}
	if (!model) {
		fprintf(stderr, "Unable to correctly read %s. Exiting\n", argv[1]);
		return 1;
	}
	int verify_model = model_check_configuration(model, vbx_cnn);
	if (verify_model == -1) {
		printf("Model %s version mismatch. Please generate the model with appropriate version of VBX_SDK \n", argv[1]);
		return 1;
	} else if (verify_model == -2) {
		printf("Model %s and VBX_CORE compression configuration mismatch. Make sure the compression configuration is set properly when the model is generated. \n", argv[1]);
		return 1;
	} else if (verify_model == -3) {
		printf("Model %s and VBX_CORE size configuration mismatch. Make sure the size configuration is set properly when the model is generated. \n", argv[1]);
		return 1;
	}

#ifdef PDMA
	const size_t pdma_capacity = 32 * 1024 * 1024;
	uint64_t pdma_out = pdma_mmap(pdma_capacity);
	int32_t pdma_channel = pdma_ch_open();
#endif

	void* read_buffer = NULL;
	vbx_cnn_io_ptr_t io_buffers[MAX_IO_BUFFERS];
	for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
		io_buffers[i] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(vbx_cnn, model_get_input_length(model, i) * sizeof(uint8_t), 1);
		if (!io_buffers[i]) {
			fprintf(stderr, "Model io_buffer requested exceeds buffer length.\n");
			return 1;
		}
	}

	unsigned expected_checksum = 0;
	bool use_test_data = std::string(argv[2]) == "TEST_DATA";
	if (!use_test_data) {
		printf("Reading %s\n", argv[2]);
		for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
			int* input_shape = model_get_input_shape(model, i);
			int input_length = model_get_input_length(model, i);
			int dims = model_get_input_dims(model, i);
			uint8_t* input_buffer = (uint8_t*)vbx_allocate_dma_buffer(vbx_cnn, input_length * sizeof(uint8_t), 0);
			if (!input_buffer) {
				fprintf(stderr, "Input_buffer requested exceeds buffer length.\n");
				return 1;
			}
			read_buffer = read_image(argv[2], input_shape[dims - 3], input_shape[dims - 2], input_shape[dims - 1], 0);
			memcpy(input_buffer, read_buffer, input_length);
			io_buffers[i] = (vbx_cnn_io_ptr_t)input_buffer;
		}
	} else {
		for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
			io_buffers[i] = (vbx_cnn_io_ptr_t)(uint8_t*)model_get_test_input(model, i);
		}
		for (unsigned o = 0; o < model_get_num_outputs(model); ++o) {
			int output_bytes = calc_type_bytes(model_get_output_datatype(model, o));
			unsigned part = fletcher32((uint16_t*)model_get_test_output(model, o),
					model_get_output_length(model, o) * output_bytes / sizeof(uint16_t));
			expected_checksum = (o == 0) ? part : (expected_checksum ^ part);
		}
	}

	unsigned num_outputs = model_get_num_outputs(model);
	for (unsigned o = 0; o < num_outputs; ++o) {
		if (vbx_cnn->comp_config == 2) {
			uint32_t output_length = model_get_output_length(model, o);
			unsigned j;
			uint32_t output_offset = 0;
			for (j = 0; j < num_outputs; j++) {
				uint32_t output_size = *((uint32_t*)model_header + 7 + (2 * j));
				output_offset = *((uint32_t*)model_header + 8 + (2 * j));
				if (output_length == output_size) {
					break;
				}
			}
			io_buffers[model_get_num_inputs(model) + o] = (vbx_cnn_io_ptr_t)((uint32_t)(uintptr_t)model + output_offset);
		} else {
			io_buffers[model_get_num_inputs(model) + o] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(
					vbx_cnn, model_get_output_length(model, o) * sizeof(uint32_t), 0);
			if (!io_buffers[model_get_num_inputs(model) + o]) {
				fprintf(stderr, "Model io_buffer requested exceeds buffer length.\n");
				return 1;
			}
			memset((void*)(io_buffers[model_get_num_inputs(model) + o]), 0,
					(size_t)(model_get_output_length(model, o) * sizeof(uint32_t)));
		}
	}

#if USE_INTERRUPTS
	enable_interrupt(vbx_cnn);
#endif
	printf("Starting inference runs\n");
	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	for (int run = 0; run < NUM_LOOPS; ++run) {
		int status;
		if (vbx_cnn->comp_config == 2) {
			status = vbx_tsnp_model_start(vbx_cnn, model, tsnp_model, input_offset, io_buffers);
		} else {
			status = vbx_cnn_model_start(vbx_cnn, model, io_buffers);
		}
#if USE_INTERRUPTS
		status = vbx_cnn_model_wfi(vbx_cnn);
#else
		while (vbx_cnn_model_poll(vbx_cnn) > 0);
#endif
		if (status < 0) {
			printf("Model failed with error %d\n", vbx_cnn_get_error_val(vbx_cnn));
		}
	}
	gettimeofday(&tv2, NULL);
	printf("network took %3.4f ms (%3.2f FPS) over %d cycles\n",
			gettimediff_us(tv1, tv2) * 1.0 / 1000 / NUM_LOOPS,
			1000. / (gettimediff_us(tv1, tv2) * 1.0 / 1000 / NUM_LOOPS), NUM_LOOPS);
#if DEBUG_CYCLE
	vbx_cnn_get_and_print_cycle_counters(vbx_cnn);
#endif

	std::vector<uint8_t*> cpu_outputs(num_outputs, NULL);
#ifdef PDMA
	size_t pdma_used = 0;
	for (unsigned o = 0; o < num_outputs; ++o) {
		size_t nbytes = model_get_output_length(model, o) * (size_t)calc_type_bytes(model_get_output_datatype(model, o));
		size_t aligned = (nbytes + 3u) & ~size_t{3};
		if (pdma_used + aligned > pdma_capacity) {
			fprintf(stderr, "Accelerator outputs (%zu bytes) exceed the PDMA buffer.\n", pdma_used + aligned);
			return 1;
		}
		memset((uint8_t*)pdma_mmap_t + pdma_used, 0, aligned);
		if (pdma_ch_transfer(pdma_out, (void*)io_buffers[model_get_num_inputs(model) + o],
					(int)pdma_used, (int)nbytes, vbx_cnn, pdma_channel) != 0) {
			fprintf(stderr, "PDMA copy of output %u failed.\n", o);
			return 1;
		}
		cpu_outputs[o] = (uint8_t*)pdma_mmap_t + pdma_used;
		pdma_used += aligned;
	}
#else
	for (unsigned o = 0; o < num_outputs; ++o) {
		cpu_outputs[o] = (uint8_t*)io_buffers[model_get_num_inputs(model) + o];
	}
#endif

	unsigned checksum = 0;
	for (unsigned o = 0; o < num_outputs; ++o) {
		int output_bytes = calc_type_bytes(model_get_output_datatype(model, o));
		int size_of_output_in_bytes = model_get_output_length(model, o) * output_bytes;
		size_of_output_in_bytes += (size_of_output_in_bytes % (int)sizeof(uint16_t));
		unsigned part = fletcher32((uint16_t*)cpu_outputs[o], size_of_output_in_bytes / (int)sizeof(uint16_t));
		checksum = (o == 0) ? part : (checksum ^ part);
	}
	printf("CHECKSUM = %08x\n", checksum);
	if (use_test_data && expected_checksum != checksum) {
		printf("Checksum mismatch for the model %s: expected = %08x, actual = %08x\n",
				argv[1], expected_checksum, checksum);
	}

	try {
		run_onnx_head(argv[3], argv[2], model, cpu_outputs);
	} catch (const Ort::Exception& err) {
		fprintf(stderr, "ONNX Runtime error: %s\n", err.what());
		return 1;
	} catch (const std::exception& err) {
		fprintf(stderr, "Error: %s\n", err.what());
		return 1;
	}

	if (read_buffer) {
		free(read_buffer);
	}
	return 0;
}
