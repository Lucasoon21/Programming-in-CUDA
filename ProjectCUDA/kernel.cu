#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <random>
#include <Windows.h>
#include <chrono>
#include <thread>
#include <sys/timeb.h>
#include <thrust/sort.h>

using namespace std;

#define SMALL 100
#define MEDIUM 400
#define BIG 1000
#define MAX_NUMBER_INT 10000
#define MAX_NUMBER_FLOAT 100.0

#define THREADS 512
#define NUMBER_OF_SORTS 10



void copy_array(float* array, float* copyArray, int size) {
	for (int i = 0; i < size; i++)
		copyArray[i] = array[i];
}

void validate_array(float* testArray, float* array, int size, string nameSort) {
	string status = "BLAD!";
	float* controlArray = (float*)malloc(sizeof(float) * size);
	copy_array(array, controlArray, size);
	sort(controlArray, controlArray + size);
	for (int i = 0; i < size; i++)
		if (testArray[i] != controlArray[i])
			cout << nameSort << " \t" << status << endl;
	free(controlArray);
}

/*==========================================================================================*/

void copy_array(int* array, int* copyArray, int size) {
	for (int i = 0; i < size; i++)
		copyArray[i] = array[i];
}

void validate_array(int* testArray, int* array, int size, string nameSort) {
	string status = "BLAD!";
	int* controlArray = (int*)malloc(sizeof(int) * size);
	copy_array(array, controlArray, size);

	sort(controlArray, controlArray + size);

	for (int i = 0; i < size; i++)
		if (testArray[i] != controlArray[i])
			cout << nameSort << " \t" << status << endl;
	free(controlArray);

}

/*=======================================================================================================================================*/
/*=======================================================================================================================================*/
/*=======================================================================================================================================*/

__global__ void GPU_bubble_sort_SEQ(float* array, int size) {
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size - 1; j++)
		{
			if (array[j] > array[j + 1])
			{
				float pom = array[j + 1];
				array[j + 1] = array[j];
				array[j] = pom;
			}
		}
	}
}

__global__ void GPU_bubble_sort_odd_PAR(float* array, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid % 2 == 1) {
		while (tid + 1 < size) {
			if (array[tid] > array[tid + 1]) {
				float tmp = array[tid];
				array[tid] = array[tid + 1];
				array[tid + 1] = tmp;
			}
			tid += gridDim.x * blockDim.x;
		}
	}
}

__global__ void GPU_bubble_sort_even_PAR(float* array, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid % 2 == 0) {
		while (tid + 1 < size) {
			if (array[tid] > array[tid + 1]) {
				float tmp = array[tid];
				array[tid] = array[tid + 1];
				array[tid + 1] = tmp;
			}

			tid += gridDim.x * blockDim.x;
		}
	}
}

void CPU_bubble_sort(float* array, int size) {
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size - 1; j++)
		{
			if (array[j] > array[j + 1])
			{
				float pom = array[j + 1];
				array[j + 1] = array[j];
				array[j] = pom;
			}
		}
	}
}

void bubble_sort(float* array, int size, float* arrayTimeSort) {
	float* array_GPU_SEQ = (float*)malloc(sizeof(float) * size);
	float* array_GPU_PAR = (float*)malloc(sizeof(float) * size);
	float* CPU_array = (float*)malloc(sizeof(float) * size);
	copy_array(array, array_GPU_SEQ, size);
	copy_array(array, array_GPU_PAR, size);
	copy_array(array, CPU_array, size);

	float* dev_seq;
	float* dev_par;

	float time_SEQ, time_PAR;
	cudaEvent_t start_SEQ, stop_SEQ, start_PAR, stop_PAR;

	//=================================================================
	cudaMalloc((void**)&dev_seq, size * sizeof(dev_seq[0]));
	cudaMemcpy(dev_seq, array_GPU_SEQ, size * sizeof(array_GPU_SEQ[0]), cudaMemcpyHostToDevice);

	cudaEventCreate(&start_SEQ);
	cudaEventCreate(&stop_SEQ);
	cudaEventRecord(start_SEQ, 0);

	GPU_bubble_sort_SEQ << <1, 1 >> > (dev_seq, size);

	cudaEventRecord(stop_SEQ, 0);
	cudaEventSynchronize(stop_SEQ);
	cudaEventElapsedTime(&time_SEQ, start_SEQ, stop_SEQ);
	cudaMemcpy(array_GPU_SEQ, dev_seq, size * sizeof(dev_seq[0]), cudaMemcpyDeviceToHost);

	//=================================================================

	cudaMalloc((void**)&dev_par, size * sizeof(dev_par[0]));
	cudaMemcpy(dev_par, array_GPU_PAR, size * sizeof(array_GPU_PAR[0]), cudaMemcpyHostToDevice);

	cudaEventCreate(&start_PAR);
	cudaEventCreate(&stop_PAR);
	cudaEventRecord(start_PAR, 0);

	int thread_per_block = THREADS;
	int blocks = (size + thread_per_block - 1) / thread_per_block;
	for (int i = 0; i < size; i++) {
		if (i % 2 == 1) {
			GPU_bubble_sort_odd_PAR << <blocks, THREADS >> > (dev_par, size);
		}
		else {
			GPU_bubble_sort_even_PAR << <blocks, THREADS >> > (dev_par, size);
		}
	}

	cudaEventRecord(stop_PAR, 0);
	cudaEventSynchronize(stop_PAR);
	cudaEventElapsedTime(&time_PAR, start_PAR, stop_PAR);

	cudaMemcpy(array_GPU_PAR, dev_par, size * sizeof(dev_par[0]), cudaMemcpyDeviceToHost);
	//=================================================================

	//printf("BUBBLE SORT \t GPU- \t %f ms \t %s \t SEKWENCYJNY\n", time_SEQ, validate_array(array_GPU_SEQ, array, size));
	validate_array(array_GPU_SEQ, array, size, "BUBBLE SORT \t GPU \tSEKWENCYJNY");
	validate_array(array_GPU_PAR, array, size, "BUBBLE SORT \t GPU \tROWNOLEGLY");

	//printf("BUBBLE SORT \t GPU- \t %f ms \t %s \t ROWNOLEGLY\n", time_PAR, validate_array(array_GPU_PAR, array, size));

	arrayTimeSort[0] += time_SEQ;
	arrayTimeSort[1] += time_PAR;

	//=================================================================

	cudaEventDestroy(start_SEQ);
	cudaEventDestroy(start_PAR);
	cudaEventDestroy(stop_SEQ);
	cudaEventDestroy(stop_PAR);

	//=================================================================

	auto start = chrono::steady_clock::now();
	CPU_bubble_sort(CPU_array, size);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;

	//printf("BUBBLE SORT \t CPU  \t %f ms \t %s\n", chrono::duration <double, milli >(diff).count(), validate_array(CPU_array, array, size));
	validate_array(CPU_array, array, size, "BUBBLE SORT \tCPU");
	arrayTimeSort[2] += chrono::duration <double, milli >(diff).count();

	cudaFree(dev_seq);
	cudaFree(dev_par);
	free(array_GPU_SEQ);
	free(array_GPU_PAR);
	free(CPU_array);
}

/*==========================================================================================*/
__global__ void GPU_bubble_sort_SEQ(int* array, int size) {
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size - 1; j++)
		{
			if (array[j] > array[j + 1])
			{
				int pom = array[j + 1];
				array[j + 1] = array[j];
				array[j] = pom;
			}
		}
	}
}

__global__ void GPU_bubble_sort_odd_PAR(int* array, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid % 2 == 1) {
		while (tid + 1 < size) {
			if (array[tid] > array[tid + 1]) {
				int tmp = array[tid];
				array[tid] = array[tid + 1];
				array[tid + 1] = tmp;
			}
			tid += gridDim.x * blockDim.x;
		}
	}
}

__global__ void GPU_bubble_sort_even_PAR(int* array, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid % 2 == 0) {
		while (tid + 1 < size) {
			if (array[tid] > array[tid + 1]) {
				int tmp = array[tid];
				array[tid] = array[tid + 1];
				array[tid + 1] = tmp;
			}

			tid += gridDim.x * blockDim.x;
		}
	}
}

void CPU_bubble_sort(int* array, int size) {
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size - 1; j++)
		{
			if (array[j] > array[j + 1])
			{
				int pom = array[j + 1];
				array[j + 1] = array[j];
				array[j] = pom;
			}
		}
	}
}

void bubble_sort(int* array, int size, float* arrayTimeSort) {
	int* array_GPU_SEQ = (int*)malloc(sizeof(int) * size);
	int* array_GPU_PAR = (int*)malloc(sizeof(int) * size);
	int* CPU_array = (int*)malloc(sizeof(int) * size);
	copy_array(array, array_GPU_SEQ, size);
	copy_array(array, array_GPU_PAR, size);
	copy_array(array, CPU_array, size);

	int* dev_seq;
	int* dev_par;

	float time_SEQ, time_PAR;
	cudaEvent_t start_SEQ, stop_SEQ, start_PAR, stop_PAR;

	//=================================================================
	cudaMalloc((void**)&dev_seq, size * sizeof(dev_seq[0]));
	cudaMemcpy(dev_seq, array_GPU_SEQ, size * sizeof(array_GPU_SEQ[0]), cudaMemcpyHostToDevice);

	cudaEventCreate(&start_SEQ);
	cudaEventCreate(&stop_SEQ);
	cudaEventRecord(start_SEQ, 0);

	GPU_bubble_sort_SEQ << <1, 1 >> > (dev_seq, size);

	cudaEventRecord(stop_SEQ, 0);
	cudaEventSynchronize(stop_SEQ);
	cudaEventElapsedTime(&time_SEQ, start_SEQ, stop_SEQ);
	cudaMemcpy(array_GPU_SEQ, dev_seq, size * sizeof(dev_seq[0]), cudaMemcpyDeviceToHost);

	//=================================================================

	cudaMalloc((void**)&dev_par, size * sizeof(dev_par[0]));
	cudaMemcpy(dev_par, array_GPU_PAR, size * sizeof(array_GPU_PAR[0]), cudaMemcpyHostToDevice);

	cudaEventCreate(&start_PAR);
	cudaEventCreate(&stop_PAR);
	cudaEventRecord(start_PAR, 0);
	int thread_per_block = THREADS;
	int blocks = (size + thread_per_block - 1) / thread_per_block;
	for (int i = 0; i < size; i++) {
		if (i % 2 == 1) {
			GPU_bubble_sort_odd_PAR << <blocks, THREADS >> > (dev_par, size);
		}
		else {
			GPU_bubble_sort_even_PAR << <blocks, THREADS >> > (dev_par, size);
		}
	}

	cudaEventRecord(stop_PAR, 0);
	cudaEventSynchronize(stop_PAR);
	cudaEventElapsedTime(&time_PAR, start_PAR, stop_PAR);

	cudaMemcpy(array_GPU_PAR, dev_par, size * sizeof(dev_par[0]), cudaMemcpyDeviceToHost);
	//=================================================================

	//printf("BUBBLE SORT \t GPU- \t %f ms \t %s \t SEKWENCYJNY\n", time_SEQ, validate_array(array_GPU_SEQ, array, size));
	//printf("BUBBLE SORT \t GPU- \t %f ms \t %s \t ROWNOLEGLY \n", time_PAR, validate_array(array_GPU_PAR, array, size));
	validate_array(array_GPU_SEQ, array, size, "BUBBLE SORT \t GPU \tSEKWENCYJNY");
	validate_array(array_GPU_PAR, array, size, "BUBBLE SORT \t GPU \tROWNOLEGLY");
	arrayTimeSort[0] += time_SEQ;
	arrayTimeSort[1] += time_PAR;
	//=================================================================

	cudaEventDestroy(start_SEQ);
	cudaEventDestroy(start_PAR);
	cudaEventDestroy(stop_SEQ);
	cudaEventDestroy(stop_PAR);
	cudaFree(dev_seq);
	cudaFree(dev_par);
	//=================================================================

	auto start = chrono::steady_clock::now();
	CPU_bubble_sort(CPU_array, size);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;

	//printf("BUBBLE SORT \t CPU \t %f ms \t %s\n", chrono::duration <double, milli >(diff).count(), validate_array(CPU_array, array, size));
	validate_array(CPU_array, array, size, "BUBBLE SORT \tCPU");

	arrayTimeSort[2] += chrono::duration <double, milli >(diff).count();
	free(array_GPU_SEQ);
	free(array_GPU_PAR);
	free(CPU_array);
}

/*=======================================================================================================================================*/
/*=======================================================================================================================================*/
/*=======================================================================================================================================*/


void CPU_quick_sort(float* array, int first, int last) {
	int pivot, j, i;
	float tmp;
	if (first < last) {
		pivot = first;
		i = first;
		j = last;
		while (i < j) {
			while (array[i] <= array[pivot] && i < last)
				i++;
			while (array[j] > array[pivot])
				j--;
			if (i < j) {
				tmp = array[i];
				array[i] = array[j];
				array[j] = tmp;
			}
		}
		tmp = array[pivot];
		array[pivot] = array[j];
		array[j] = tmp;
		CPU_quick_sort(array, first, j - 1);
		CPU_quick_sort(array, j + 1, last);
	}
}


void quick_sort(float* array, int size, float* arrayTimeSort) {
	float* array_CPU = (float*)malloc(sizeof(float) * size);
	copy_array(array, array_CPU, size);

	

	auto start = chrono::steady_clock::now();
	CPU_quick_sort(array_CPU, 0, size - 1);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;

	//	printf("QUICK SORT \t CPU \t %f ms \t %s\n", chrono::duration <double, milli >(diff).count(), validate_array(array_CPU, array, size));
	validate_array(array_CPU, array, size, "QUICK SORT \tCPU");

	arrayTimeSort[3] += chrono::duration <double, milli >(diff).count();
	free(array_CPU);
}



/*==========================================================================================*/

void CPU_quick_sort(int* array, int first, int last) {
	int pivot, j, i;
	int tmp;
	if (first < last) {
		pivot = first;
		i = first;
		j = last;
		while (i < j) {
			while (array[i] <= array[pivot] && i < last)
				i++;
			while (array[j] > array[pivot])
				j--;
			if (i < j) {
				tmp = array[i];
				array[i] = array[j];
				array[j] = tmp;
			}
		}
		tmp = array[pivot];
		array[pivot] = array[j];
		array[j] = tmp;
		CPU_quick_sort(array, first, j - 1);
		CPU_quick_sort(array, j + 1, last);
	}
}

void quick_sort(int* array, int size, float* arrayTimeSort)
{
	int* array_CPU = (int*)malloc(sizeof(int) * size);
	copy_array(array, array_CPU, size);


	auto start = chrono::steady_clock::now();
	CPU_quick_sort(array_CPU, 0, size - 1);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;

	//printf("QUICK SORT \t CPU \t %f ms \t %s\n", chrono::duration <double, milli >(diff).count(), validate_array(array_CPU, array, size));
	validate_array(array_CPU, array, size, "QUICK SORT \tCPU");

	arrayTimeSort[3] += chrono::duration <double, milli >(diff).count();
	free(array_CPU);
}

/*=======================================================================================================================================*/
/*=======================================================================================================================================*/
/*=======================================================================================================================================*/

void CPU_rank_sort(float* a, float* b, int size) {
	for (int i = 0; i < size; i++) {
		int x = 0;
		for (int j = 0; j < size; j++)
			if (a[i] > a[j])
				x++;
		b[x] = a[i];
	}
}

__global__ void GPU_rank_sort(float* array_to_sort, int* ranks_array, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		int i;
		int rank = 0;
		for (i = 0; i < size; i++) {
			if (i != tid) {
				if (array_to_sort[tid] > array_to_sort[i]) {
					rank++;
				}
			}
		}
		ranks_array[tid] = rank;
		tid += gridDim.x * blockDim.x;
	}
}

__global__ void GPU_rs_set_ranks(float* array_to_sort, int* ranks_array, float* sorted_array, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		sorted_array[ranks_array[tid]] = array_to_sort[tid];
		tid += gridDim.x * blockDim.x;
	}
}

void rank_sort(float* tablica, int size, float* arrayTimeSort) {
	float* array = (float*)malloc(sizeof(float) * size);
	float* sorted_array = (float*)malloc(sizeof(float) * size);
	float* sorted_array_cpu = (float*)malloc(sizeof(float) * size);
	int* ranks_array = (int*)malloc(sizeof(int) * size);

	copy_array(tablica, array, size);

	float* dev_array_gpu;
	int* dev_ranks;
	float* dev_array_sorted;

	cudaMalloc((void**)&dev_array_gpu, size * sizeof(dev_array_gpu[0]));
	cudaMalloc((void**)&dev_ranks, size * sizeof(dev_ranks[0]));
	cudaMalloc((void**)&dev_array_sorted, size * sizeof(dev_array_sorted[0]));

	cudaMemcpy(dev_array_gpu, array, size * sizeof(array[0]), cudaMemcpyHostToDevice);
	//=================================================================
	float time_GPU;
	cudaEvent_t start_GPU, stop_GPU;

	cudaEventCreate(&start_GPU);
	cudaEventCreate(&stop_GPU);

	cudaEventRecord(start_GPU, 0);
	int thread_per_block = THREADS;
	int blocks = (size + thread_per_block - 1) / thread_per_block;
	GPU_rank_sort << <blocks, THREADS >> > (dev_array_gpu, dev_ranks, size);
	GPU_rs_set_ranks << <blocks, THREADS >> > (dev_array_gpu, dev_ranks, dev_array_sorted, size);

	cudaEventRecord(stop_GPU, 0);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU);
	cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);
	//=================================================================

	cudaMemcpy(sorted_array, dev_array_sorted, size * sizeof(dev_array_sorted[0]), cudaMemcpyDeviceToHost);
	cudaMemcpy(array, dev_array_gpu, size * sizeof(dev_array_gpu[0]), cudaMemcpyDeviceToHost);

	//	printf("RANK SORT \t GPU- \t %f ms \t %s\n", time_GPU, validate_array(sorted_array, tablica, size));
	validate_array(sorted_array, array, size, "RANK SORT \tGPU");

	arrayTimeSort[4] += time_GPU;
	cudaFree(dev_array_gpu);
	cudaFree(dev_ranks);
	cudaFree(dev_array_sorted);

	copy_array(tablica, array, size);


	auto start = chrono::steady_clock::now();

	CPU_rank_sort(array, sorted_array_cpu, size);

	auto end = chrono::steady_clock::now();
	auto diff = end - start;

	//	printf("RANK SORT \t CPU \t %f ms \t %s\n", chrono::duration <double, milli >(diff).count(), validate_array(sorted_array_cpu, tablica, size));
	validate_array(sorted_array_cpu, array, size, "RANK SORT \tCPU");

	arrayTimeSort[5] += chrono::duration <double, milli >(diff).count();

	free(sorted_array_cpu);
	free(ranks_array);
	free(array);
	free(sorted_array);
}

/*==========================================================================================*/
void CPU_rank_sort(int* a, int* b, int size) {
	for (int i = 0; i < size; i++) {
		int x = 0;
		for (int j = 0; j < size; j++)
			if (a[i] > a[j])
				x++;
		b[x] = a[i];
	}
}

__global__ void GPU_rank_sort(int* array_to_sort, int* ranks_array, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		int i;
		int rank = 0;
		for (i = 0; i < size; i++) {
			if (i != tid) {
				if (array_to_sort[tid] > array_to_sort[i]) {
					rank++;
				}
			}
		}
		ranks_array[tid] = rank;
		tid += gridDim.x * blockDim.x;
	}
}

__global__ void GPU_rs_set_ranks(int* array_to_sort, int* ranks_array, int* sorted_array, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		sorted_array[ranks_array[tid]] = array_to_sort[tid];
		tid += gridDim.x * blockDim.x;
	}
}

void rank_sort(int* array, int size, float* arrayTimeSort) {
	int* array_CPU = (int*)malloc(sizeof(int) * size);
	int* sorted_array = (int*)malloc(sizeof(int) * size);
	int* sorted_array_cpu = (int*)malloc(sizeof(int) * size);
	int* ranks_array = (int*)malloc(sizeof(int) * size);

	copy_array(array, array_CPU, size);

	int* dev_array_gpu;
	int* dev_ranks;
	int* dev_array_sorted;

	cudaMalloc((void**)&dev_array_gpu, size * sizeof(dev_array_gpu[0]));
	cudaMalloc((void**)&dev_ranks, size * sizeof(dev_ranks[0]));
	cudaMalloc((void**)&dev_array_sorted, size * sizeof(dev_array_sorted[0]));

	cudaMemcpy(dev_array_gpu, array_CPU, size * sizeof(array_CPU[0]), cudaMemcpyHostToDevice);
	//=================================================================
	float time_GPU;
	cudaEvent_t start_GPU, stop_GPU;

	cudaEventCreate(&start_GPU);
	cudaEventCreate(&stop_GPU);
	cudaEventRecord(start_GPU, 0);
	int thread_per_block = THREADS;
	int blocks = (size + thread_per_block - 1) / thread_per_block;
	GPU_rank_sort << <blocks, THREADS >> > (dev_array_gpu, dev_ranks, size);
	GPU_rs_set_ranks << <blocks, THREADS >> > (dev_array_gpu, dev_ranks, dev_array_sorted, size);

	cudaEventRecord(stop_GPU, 0);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU);
	//=================================================================

	cudaMemcpy(sorted_array, dev_array_sorted, size * sizeof(dev_array_sorted[0]), cudaMemcpyDeviceToHost);
	cudaMemcpy(array_CPU, dev_array_gpu, size * sizeof(dev_array_gpu[0]), cudaMemcpyDeviceToHost);

	cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);

	//printf("RANK SORT \t GPU- \t %f ms \t %s\n", time_GPU, validate_array(sorted_array, tablica, size));
	validate_array(sorted_array, array, size, "RANK SORT \tGPU");

	arrayTimeSort[4] += time_GPU;
	cudaFree(dev_array_gpu);
	cudaFree(dev_ranks);
	cudaFree(dev_array_sorted);

	copy_array(array, array_CPU, size);


	auto start = chrono::steady_clock::now();

	CPU_rank_sort(array_CPU, sorted_array_cpu, size);

	auto end = chrono::steady_clock::now();
	auto diff = end - start;

	//printf("RANK SORT \t CPU \t %f ms \t %s\n", chrono::duration <double, milli >(diff).count(), validate_array(sorted_array_cpu, tablica, size));
	validate_array(sorted_array_cpu, array, size, "RANK SORT \tCPU");

	arrayTimeSort[5] += chrono::duration <double, milli >(diff).count();
	free(array_CPU);
	free(sorted_array);
	free(sorted_array_cpu);
	free(ranks_array);
}

/*=======================================================================================================================================*/
/*=====================================================================================================================================*/
/*=======================================================================================================================================*/

void merge(float* array, int left, int mid, int right) {
	int i, j, k;
	int n1 = mid - left + 1;
	int n2 = right - mid;

	float* LeftArray = (float*)malloc(sizeof(float) * n1);
	float* RightArray = (float*)malloc(sizeof(float) * n1);

	for (i = 0; i < n1; i++)
		LeftArray[i] = array[left + i];
	for (j = 0; j < n2; j++)
		RightArray[j] = array[mid + 1 + j];

	i = 0;
	j = 0;
	k = left;
	while (i < n1 && j < n2) {
		if (LeftArray[i] <= RightArray[j]) {
			array[k] = LeftArray[i];
			i++;
		}
		else {
			array[k] = RightArray[j];
			j++;
		}
		k++;
	}

	while (i < n1) {
		array[k] = LeftArray[i];
		i++;
		k++;
	}

	while (j < n2) {
		array[k] = RightArray[j];
		j++;
		k++;
	}
	free(LeftArray);
	free(RightArray);
}

void CPU_merge_sort(float* array, int begin, int end) {
	if (begin < end) {
		int m = begin + (end - begin) / 2;
		CPU_merge_sort(array, begin, m);
		CPU_merge_sort(array, m + 1, end);
		merge(array, begin, m, end);
	}
}

void merge_sort(float* array, int size, float* arrayTimeSort) {
	float* array_CPU = (float*)malloc(sizeof(float) * size);
	copy_array(array, array_CPU, size);
	auto start = chrono::steady_clock::now();

	CPU_merge_sort(array_CPU, 0, size - 1);

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	//printf("MERGE SORT \t CPU \t %f ms \t %s\n", chrono::duration <double, milli >(diff).count(), validate_array(array_CPU, array, size));
	validate_array(array_CPU, array, size, "MERGE SORT \tCPU");

	arrayTimeSort[6] += chrono::duration <double, milli >(diff).count();
	free(array_CPU);

}
/*==========================================================================================*/

void merge(int* array, int left, int mid, int right) {
	int i, j, k;
	int n1 = mid - left + 1;
	int n2 = right - mid;

	int* LeftArray = (int*)malloc(sizeof(int) * n1);
	int* RightArray = (int*)malloc(sizeof(int) * n1);

	for (i = 0; i < n1; i++)
		LeftArray[i] = array[left + i];
	for (j = 0; j < n2; j++)
		RightArray[j] = array[mid + 1 + j];

	i = 0;
	j = 0;
	k = left;
	while (i < n1 && j < n2) {
		if (LeftArray[i] <= RightArray[j]) {
			array[k] = LeftArray[i];
			i++;
		}
		else {
			array[k] = RightArray[j];
			j++;
		}
		k++;
	}

	while (i < n1) {
		array[k] = LeftArray[i];
		i++;
		k++;
	}

	while (j < n2) {
		array[k] = RightArray[j];
		j++;
		k++;
	}
	free(LeftArray);
	free(RightArray);
}

void CPU_merge_sort(int* array, int begin, int end) {
	if (begin < end) {
		int mid = begin + (end - begin) / 2;
		CPU_merge_sort(array, begin, mid);
		CPU_merge_sort(array, mid + 1, end);
		merge(array, begin, mid, end);
	}
}

void merge_sort(int* array, int size, float* arrayTimeSort) {
	int* array_CPU = (int*)malloc(sizeof(int) * size);
	copy_array(array, array_CPU, size);
	auto start = chrono::steady_clock::now();
	CPU_merge_sort(array_CPU, 0, size - 1);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	//printf("MERGE SORT \t CPU \t %f ms \t %s\n", chrono::duration <double, milli >(diff).count(), validate_array(array_CPU, array, size));
	validate_array(array_CPU, array, size, "MERGE SORT \tCPU");

	arrayTimeSort[6] += chrono::duration <double, milli >(diff).count();
	free(array_CPU);

}

/*======================================================================================================================================*/
/*=======================================================================================================================================*/
/*=======================================================================================================================================*/

void GPU_thrust_sort(float* array, int size) {
	thrust::sort(array, array + size);
}

void thrust_sort(float* array, int size, float* arrayTimeSort) {
	float* array_GPU = (float*)malloc(sizeof(float) * size);
	copy_array(array, array_GPU, size);

	float time_GPU;
	cudaEvent_t start_GPU, stop_GPU;

	cudaEventCreate(&start_GPU);
	cudaEventCreate(&stop_GPU);
	cudaEventRecord(start_GPU, 0);

	GPU_thrust_sort(array_GPU, size);

	cudaEventRecord(stop_GPU, 0);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU);
	cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);

	//printf("THRUST SORT \t GPU- \t %f ms \t %s\n", time_GPU, validate_array(array_GPU, array, size));
	validate_array(array_GPU, array, size, "THRUST SORT \tGPU");

	arrayTimeSort[7] += time_GPU;
	free(array_GPU);

}

/*==========================================================================================*/

void GPU_thrust_sort(int* array, int size) {
	thrust::sort(array, array + size);
}

void thrust_sort(int* array, int size, float* arrayTimeSort) {
	int* array_GPU = (int*)malloc(sizeof(int) * size);
	copy_array(array, array_GPU, size);
	float time_GPU;
	cudaEvent_t start_GPU, stop_GPU;

	cudaEventCreate(&start_GPU);
	cudaEventCreate(&stop_GPU);
	cudaEventRecord(start_GPU, 0);

	GPU_thrust_sort(array_GPU, size);

	cudaEventRecord(stop_GPU, 0);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU);
	cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);

	//	printf("THRUST SORT \t GPU- \t %f ms \t %s\n", time_GPU, validate_array(array_GPU, array, size));
	validate_array(array_GPU, array, size, "THRUST SORT \tGPU");

	arrayTimeSort[7] += time_GPU;
	free(array_GPU);

}

/*=======================================================================================================================================*/
/*=======================================================================================================================================*/
/*=======================================================================================================================================*/

void CPU_stl_sort(float* array, int size) {
	sort(array, array + size);
}

void stl_sort(float* array, int size, float* arrayTimeSort) {
	float* array_CPU = (float*)malloc(sizeof(float) * size);
	copy_array(array, array_CPU, size);
	auto start = chrono::steady_clock::now();

	CPU_stl_sort(array_CPU, size);

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	//printf("SORT STL \t CPU \t %f ms \t %s\n\n", chrono::duration <double, milli >(diff).count(), validate_array(array_CPU, array, size));
	validate_array(array_CPU, array, size, "STL SORT \tCPU");

	arrayTimeSort[8] += chrono::duration <double, milli >(diff).count();
	free(array_CPU);

}

/*==========================================================================================*/


void CPU_stl_sort(int* array, int size) {
	sort(array, array + size);
}

void stl_sort(int* array, int size, float* arrayTimeSort) {
	int* array_CPU = (int*)malloc(sizeof(int) * size);
	copy_array(array, array_CPU, size);

	auto start = chrono::steady_clock::now();

	CPU_stl_sort(array_CPU, size);

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	//printf("SORT STL \t CPU \t %f ms \t %s\n\n", chrono::duration <double, milli >(diff).count(), validate_array(array_CPU, array, size));
	validate_array(array_CPU, array, size, "STL SORT \tCPU");

	arrayTimeSort[8] += chrono::duration <double, milli >(diff).count();
	free(array_CPU);

}

/*=======================================================================================================================================*/
/*=======================================================================================================================================*/
/*=======================================================================================================================================*/

bool check_repeat_value(int* array, int size, int value) {
	for (int i = 0; i < size; i++) {
		if (array[i] == value) {
			return true;
		}
	}
	return false;
}

bool check_repeat_value(float* array, int size, float value) {
	for (int i = 0; i < size; i++) {
		if (array[i] == value) {
			return true;
		}
	}
	return false;
}

/*==========================================================================================*/

void sort_int() {
	int listSort = 9;
	float* arrayTimeSort = (float*)malloc(sizeof(float) * listSort);
	for (int i = 0; i < listSort; i++) {
		arrayTimeSort[i] = 0;
	}

	string sortName[9];
	sortName[0] = "BUBBLE SORT SEKW \t GPU- \t";
	sortName[1] = "BUBBLE SORT ROWN \t GPU- \t";
	sortName[2] = "BUBBLE SORT \t\t CPU \t";
	sortName[3] = "QUICK SORT \t\t CPU \t";
	sortName[4] = "RANK SORT \t\t GPU- \t";
	sortName[5] = "RANK SORT \t\t CPU \t";
	sortName[6] = "MERGE SORT \t\t CPU \t";
	sortName[7] = "THRUST SORT \t\t GPU- \t";
	sortName[8] = "STL SORT \t\t CPU \t";


	srand(time(NULL));






	printf("\n============================================================================================================");
	printf("\n==================================== LICZBY CALKOWITE OD O D0 %d ========================================", MAX_NUMBER_INT);
	printf("\n============================================================================================================\n");
	printf("\n============================ \tSORTOWANIE DLA %d ELEMENTOW \t-\t INT \t============================\n", SMALL);

	for (int i = 0; i < NUMBER_OF_SORTS; i++) {

		int num_int_small[SMALL];
		for (int i = 0; i < SMALL; i++) {
			int value = rand() % MAX_NUMBER_INT;
			while (check_repeat_value(num_int_small, i, value) == true) {
				value = rand() % MAX_NUMBER_INT;
			}
			num_int_small[i] = value;
		}
		bubble_sort(num_int_small, SMALL, arrayTimeSort);
		quick_sort(num_int_small, SMALL, arrayTimeSort);
		rank_sort(num_int_small, SMALL, arrayTimeSort);
		merge_sort(num_int_small, SMALL, arrayTimeSort);
		thrust_sort(num_int_small, SMALL, arrayTimeSort);
		stl_sort(num_int_small, SMALL, arrayTimeSort);

	}
	for (int i = 0; i < listSort; i++) {
		cout << sortName[i] << arrayTimeSort[i] / NUMBER_OF_SORTS << " ms" << endl;
	}

	for (int i = 0; i < listSort; i++) {
		arrayTimeSort[i] = 0;
	}


	printf("\n\n============================ \tSORTOWANIE DLA %d ELEMENTOW \t-\t INT \t============================\n", MEDIUM);
	for (int i = 0; i < NUMBER_OF_SORTS; i++) {

		int num_int_medium[MEDIUM];
		for (int i = 0; i < MEDIUM; i++) {
			int value = rand() % MAX_NUMBER_INT;
			while (check_repeat_value(num_int_medium, i, value) == true) {
				value = rand() % MAX_NUMBER_INT;
			}
			num_int_medium[i] = value;
		}
		bubble_sort(num_int_medium, MEDIUM, arrayTimeSort);
		quick_sort(num_int_medium, MEDIUM, arrayTimeSort);
		rank_sort(num_int_medium, MEDIUM, arrayTimeSort);
		merge_sort(num_int_medium, MEDIUM, arrayTimeSort);
		thrust_sort(num_int_medium, MEDIUM, arrayTimeSort);
		stl_sort(num_int_medium, MEDIUM, arrayTimeSort);

	}
	for (int i = 0; i < listSort; i++) {
		cout << sortName[i] << arrayTimeSort[i] / NUMBER_OF_SORTS << " ms" << endl;
	}



	for (int i = 0; i < listSort; i++) {
		arrayTimeSort[i] = 0;
	}


	printf("\n\n============================ \tSORTOWANIE DLA %d ELEMENTOW \t-\t INT \t============================\n", BIG);
	for (int i = 0; i < NUMBER_OF_SORTS; i++) {

		int num_int_big[BIG];
		for (int i = 0; i < BIG; i++) {
			int value = rand() % MAX_NUMBER_INT;
			while (check_repeat_value(num_int_big, i, value) == true) {
				value = rand() % MAX_NUMBER_INT;
			}
			num_int_big[i] = value;
		}
		bubble_sort(num_int_big, BIG, arrayTimeSort);
		quick_sort(num_int_big, BIG, arrayTimeSort);
		rank_sort(num_int_big, BIG, arrayTimeSort);
		merge_sort(num_int_big, BIG, arrayTimeSort);
		thrust_sort(num_int_big, BIG, arrayTimeSort);
		stl_sort(num_int_big, BIG, arrayTimeSort);
	}
	for (int i = 0; i < listSort; i++) {
		cout << sortName[i] << arrayTimeSort[i] / NUMBER_OF_SORTS << " ms" << endl;
	}
	free(arrayTimeSort);

}

void sort_float() {
	srand(time(NULL));


	int listSort = 9;
	float* arrayTimeSort = (float*)malloc(sizeof(float) * listSort);
	for (int i = 0; i < listSort; i++) {
		arrayTimeSort[i] = 0;
	}


	string sortName[9];
	sortName[0] = "BUBBLE SORT SEKW \t GPU- \t";
	sortName[1] = "BUBBLE SORT ROWN \t GPU- \t";
	sortName[2] = "BUBBLE SORT \t\t CPU \t";
	sortName[3] = "QUICK SORT \t\t CPU \t";
	sortName[4] = "RANK SORT \t\t GPU- \t";
	sortName[5] = "RANK SORT \t\t CPU \t";
	sortName[6] = "MERGE SORT \t\t CPU \t";
	sortName[7] = "THRUST SORT \t\t GPU- \t";
	sortName[8] = "STL SORT \t\t CPU \t";

	printf("\n##############################################################################################################");
	printf("\n#------------------------------------------------------------------------------------------------------------#");
	printf("\n#---------------------------- LICZBY ZMIENNOPRZECINKOWE OD 0 DO %f ----------------------------------#", MAX_NUMBER_FLOAT);
	printf("\n#------------------------------------------------------------------------------------------------------------#");
	printf("\n##############################################################################################################\n");



	printf("\n============================ \tSORTOWANIE DLA %d ELEMENTOW \t-\t FLOAT \t============================\n", SMALL);
	for (int i = 0; i < NUMBER_OF_SORTS; i++) {


		float num_float_small[SMALL];
		for (int i = 0; i < SMALL; i++) {
			float value = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_NUMBER_FLOAT));
			while (check_repeat_value(num_float_small, i, value) == true) {
				value = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_NUMBER_FLOAT));
			}
			num_float_small[i] = value;
		}
		bubble_sort(num_float_small, SMALL, arrayTimeSort);
		quick_sort(num_float_small, SMALL, arrayTimeSort);
		rank_sort(num_float_small, SMALL, arrayTimeSort);
		merge_sort(num_float_small, SMALL, arrayTimeSort);
		thrust_sort(num_float_small, SMALL, arrayTimeSort);
		stl_sort(num_float_small, SMALL, arrayTimeSort);
	}


	for (int i = 0; i < listSort; i++) {
		cout << sortName[i] << arrayTimeSort[i] / NUMBER_OF_SORTS << " ms" << endl;
	}

	for (int i = 0; i < listSort; i++) {
		arrayTimeSort[i] = 0;
	}





	printf("\n\n============================ \tSORTOWANIE DLA %d ELEMENTOW \t-\t FLOAT \t============================\n", MEDIUM);
	for (int i = 0; i < NUMBER_OF_SORTS; i++) {
		float num_flaot_medium[MEDIUM];

		for (int i = 0; i < MEDIUM; i++) {
			float value = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_NUMBER_FLOAT));
			while (check_repeat_value(num_flaot_medium, i, value) == true) {
				value = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_NUMBER_FLOAT));
			}
			num_flaot_medium[i] = value;
		}
		bubble_sort(num_flaot_medium, MEDIUM, arrayTimeSort);
		quick_sort(num_flaot_medium, MEDIUM, arrayTimeSort);
		rank_sort(num_flaot_medium, MEDIUM, arrayTimeSort);
		merge_sort(num_flaot_medium, MEDIUM, arrayTimeSort);
		thrust_sort(num_flaot_medium, MEDIUM, arrayTimeSort);
		stl_sort(num_flaot_medium, SMALL, arrayTimeSort);

	}



	for (int i = 0; i < listSort; i++) {
		cout << sortName[i] << arrayTimeSort[i] / NUMBER_OF_SORTS << " ms" << endl;
	}

	for (int i = 0; i < listSort; i++) {
		arrayTimeSort[i] = 0;
	}




	printf("\n\n============================ \tSORTOWANIE DLA %d ELEMENTOW \t-\t FLOAT \t============================\n", BIG);
	for (int i = 0; i < NUMBER_OF_SORTS; i++) {
		float num_float_big[BIG];

		for (int i = 0; i < BIG; i++) {
			float value = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_NUMBER_FLOAT));
			while (check_repeat_value(num_float_big, i, value) == true) {
				value = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_NUMBER_FLOAT));
			}
			num_float_big[i] = value;
		}
		bubble_sort(num_float_big, BIG, arrayTimeSort);
		quick_sort(num_float_big, BIG, arrayTimeSort);
		rank_sort(num_float_big, BIG, arrayTimeSort);
		merge_sort(num_float_big, BIG, arrayTimeSort);
		thrust_sort(num_float_big, BIG, arrayTimeSort);
		stl_sort(num_float_big, SMALL, arrayTimeSort);
	}
	for (int i = 0; i < listSort; i++) {
		cout << sortName[i] << arrayTimeSort[i] / NUMBER_OF_SORTS << " ms" << endl;
	}

	for (int i = 0; i < listSort; i++) {
		arrayTimeSort[i] = 0;
	}
	free(arrayTimeSort);
}

/*==========================================================================================*/

int main()
{
	cout << "OBJASNIENIA:" << endl;
	cout << "1. Kazde sortowanie posiada taka sama tablice" << endl;
	cout << "2. Jesli tablica bedzie blednie posortowana to pojawi sie komunikat (przykladowa tresc ponizej). Sprawdzana tablica jest porownywana z tablica ktora zostala posortowana za pomoca gotowej funkcji std::sort" << endl;
	cout << "BUBBLE SORT \t BLAD!" << endl;
	cout << "3. w konsoli pokazany jest sredni czas sortowania " << NUMBER_OF_SORTS << " tablic" << endl;
	cout << "4. Sortowanie odbywa sie dla liczb calkowityhc (INT) oraz zmiennoprzecinkowych (FLOAT)" << endl;
	cout << "5. Sortowane sa tablica o 3 rozmiarach: " << SMALL << ", " << MEDIUM << " oraz " << BIG << endl;


	sort_int();
	sort_float();

	return 0;
}