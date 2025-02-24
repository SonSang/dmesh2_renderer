#pragma once

#include <iostream>
#include <vector>
#include "renderer.h"
#include <cuda_runtime_api.h>

namespace CudaRenderer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct FaceState
	{
		size_t scan_size;
		char* scanning_space;			// used to compute [face_offsets] later;
		float* depths;
		float* min_depths;
		float* max_depths;
		uint32_t* tiles_touched;
		uint32_t* face_offsets; 

		static FaceState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* final_T;
		float* final_prev_T;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* face_list_keys_unsorted;
		uint64_t* face_list_keys;
		uint32_t* face_list_unsorted;
		uint32_t* face_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t F);
	};

	struct ImageRenderLayerState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		int* first_face;
		int* first_tet;

		static ImageRenderLayerState fromChunk(char*& chunk, size_t N);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};