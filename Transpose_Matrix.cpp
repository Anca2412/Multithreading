// LabClass.cpp : Defines the entry point for the console application.
//
//My tests on an Intel i7-4712MQ, 8 cores
/*
	N=512
	Normal:				269 ms								
	Multithreading:		68 ms									
	Multithreading												
		& line access:  26 ms
	SIMD1:				18 ms
	SIMD2:				20 ms	

	N=2048   
	Normal:				84 s
	Multithreading:		42 s
	Multithreading 
		& line access:  2,5 s
	SIMD1:				0,9 s
	SIMD2:				0,9 s

	Conclusion: 

	Once the buffer is overloaded the processing time increases considerably. Reason is that CPU waits for memory 
	requests to be fulfilled and this leads to memory latency. Every access of data which is not in the cache is more expensive in terms of time
	because RAM or sometimes even HDD needs to be accesed. Then data present in the cache will always be easily and fast accesible.
	Therefore time will not be slowed down by a factor of 64 because its variance depends on available resources 
	and amount of information needed by the CPU which can be partially accessed from cache and partially from RAM. 
	Exact procentage and numbers always depend on the arhitechture of the operating system and the free resources available on runtime.

	-----

	Extra test: Accessing the matrix in a more efficient way of storing data in the cache by first accesing line i and column k of first
				matrix and then the coresponding elements of second matrix (last 2 for loops are switched)

	N 512, Multithreading with line and column access: 39 ms
	N 2048, Multithreading with line and column access: 2 s

	Conclusion: Using the memory properly can significantly improove performance. The results can even be similar to  SIMD or line access.
*/

#include "stdafx.h"
#include <pthread.h>
#include <process.h>  
#include <thread>
#include <atomic>
#include <mutex>
#include <intrin.h>
#include "Chrono.h"
#include <iostream>
// lab1MatMul.cpp : Defines the entry point for the console application.
//


#define N 2048
#define nbThreads 4
float *A, *B,*C, *D, *P;
float *AUnaligned, *BUnaligned,*CUnaligned, *DUnaligned, *PUnaligned;

void AllocateAndPopulate()  //Allocate arrays A,B and C for the three arrays and make sure that pointers are 32-byte aligned!
{
	AUnaligned=A=new float[N*N+16];
	BUnaligned=B=new float[N*N+16];
	CUnaligned=C=new float[N*N+16];
	DUnaligned=D=new float[N*N+16];
	PUnaligned = P = new float[N*N + 16];
	int alignA=(((unsigned long long) A) & 31)/4; //Compute the address modulo 32 (bytes), and divid it by 4 (bytes per float) to get a modulo 8 (floats)
	int alignB=(((unsigned long long) A) & 31)/4; //Compute the address modulo 32 (bytes), and divid it by 4 (bytes per float) to get a modulo 8 (floats)
	int alignC=(((unsigned long long) A) & 31)/4; //Compute the address modulo 32 (bytes), and divid it by 4 (bytes per float) to get a modulo 8 (floats)
	int alignD=(((unsigned long long) A) & 31)/4;
	int alignP = (((unsigned long long) A) & 31) / 4;
	A+=8-alignA; //Align A address. A address is now a multiple of 32 bytes (i.e., multiple of 8x4) 
	B+=8-alignB; //Align B address
	C+=8-alignC; //Align C address
	D+=8-alignD;
	P += 8 - alignP;
	for (int i=0;i<N*N;i++)
	{
		A[i]=(rand()+0.5f)/(RAND_MAX+1.f);
		B[i]=(rand()+0.5f)/(RAND_MAX+1.f);
	}
}

void DeAllocate() //deallocate A,B and C!
{
	delete[] AUnaligned;
	delete[] BUnaligned;
	delete[] CUnaligned;
	delete[] DUnaligned;
	delete[] PUnaligned;
}


#define fma8(a,b,c) (_mm256_fmadd_ps((a),(b),(c))) 
#define add8(a,b) (_mm256_add_ps((a),(b))) 
#define mul8(a,b) (_mm256_mul_ps((a),(b))) 
#define set1(a) (_mm256_set1_ps(a)) 
typedef __m256 float8;

void MatrixMult()
{
	for (int i=0;i<N;i++)
		for (int j=0;j<N;j++)
		{
			C[i*N+j]=0;
			for (int k=0;k<N;k++)
					C[i*N+j]+=A[i*N+k]*B[k*N+j];
		}
}

// Multithreading ------------------------------------------------------------
void setElementsRes(int x)
{
	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i++)
		for (int j = 0; j < N; j++)
			D[i*N + j] = 0;
	
	for (int i = x*N / nbThreads; i < (x+1)* N / nbThreads; i++)
		//for (int k = 0; k < N; k++)
			for (int j = 0; j < N; j++)
				for (int k = 0; k < N; k++)
				D[i*N + j] += A[i*N + k] * B[k*N + j];

}

void MatrixMult_Mutithreading()
{
	std::thread t[nbThreads];
	for (int i = 0; i < nbThreads; i++)
		t[i] = std::thread(setElementsRes, i);
	//Join 
	for (int i = 0; i < nbThreads; i++) {
		t[i].join();
	}
}

// Transpose Matrix ------------------------------------------------------
void transpose_Matrix()
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			P[i*N + j] = B[j*N + i];
}


void setLineTransposed(int x)
{
	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i++)
		for (int j = 0; j < N; j++)
			D[i*N + j] = 0;

	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i++)
		//for (int k = 0; k < N; k++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				D[i*N + j] += A[i*N + k] * P[j*N + k];

}

void MatrixMult_Transposed()
{
	std::thread t[nbThreads];
	for (int i = 0; i < nbThreads; i++)
		t[i] = std::thread(setLineTransposed, i);
	//Join 
	for (int i = 0; i < nbThreads; i++) {
		t[i].join();
	}
}

// Submatrix & SIMD 1 ---------------------------------------------------------------
void setElementsSIMD1(int x)
{
	float8 temp;
	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i++)
		for (int j = 0; j < N; j += 8)
		{
			temp = set1(0);
			_mm256_store_ps(&D[i*N + j], temp);
		}			

	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i++)
		
			for (int j = 0; j < N; j ++)
				for (int k = 0; k < N; k += 16)
			{				
				temp = set1(0);
				temp = add8(temp, mul8(_mm256_load_ps(&A[i*N + k]), _mm256_load_ps(&P[j*N + k])));
				/*D[i*N + j] += temp.m256_f32[0] + temp.m256_f32[1] + temp.m256_f32[2] + temp.m256_f32[3] + 
							 temp.m256_f32[4] +  temp.m256_f32[5] + temp.m256_f32[6] + temp.m256_f32[7];

				temp = set1(0);*/
				temp = add8(temp, mul8(_mm256_load_ps(&A[i*N + k+8]), _mm256_load_ps(&P[j*N + k+8])));
				D[i*N + j] += temp.m256_f32[0] + temp.m256_f32[1] + temp.m256_f32[2] + temp.m256_f32[3] +
					temp.m256_f32[4] + temp.m256_f32[5] + temp.m256_f32[6] + temp.m256_f32[7];
			}
}

void MatrixMult_SIMD1()
{
	std::thread t[nbThreads];
	for (int i = 0; i < nbThreads; i++)
		t[i] = std::thread(setElementsSIMD1, i);
	//Join 
	for (int i = 0; i < nbThreads; i++) {
		t[i].join();
	}
}

// SIMD 2 ---------------------------------------------------------------
void setElementsSIMD2(int x)
{
	float8 temp;
	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i++)
		for (int j = 0; j < N; j += 8)
		{
			temp = set1(0);
			_mm256_store_ps(&D[i*N + j], temp);
			//_mm256_store_ps(&D[i*N + j+8], temp);
		}

	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i+=2)

		for (int j = 0; j < N; j+=2)
			for (int k = 0; k < N; k += 16)
			{
				temp = set1(0);
				temp = add8(temp, mul8(_mm256_load_ps(&A[i*N + k]), _mm256_load_ps(&P[j*N + k])));
				temp = add8(temp, mul8(_mm256_load_ps(&A[i*N + k + 8]), _mm256_load_ps(&P[j*N + k + 8])));
				D[i*N + j] += temp.m256_f32[0] + temp.m256_f32[1] + temp.m256_f32[2] + temp.m256_f32[3] +
					temp.m256_f32[4] + temp.m256_f32[5] + temp.m256_f32[6] + temp.m256_f32[7];

				temp = set1(0);
				temp = add8(temp, mul8(_mm256_load_ps(&A[(i+1)*N + k]), _mm256_load_ps(&P[j*N + k])));
				temp = add8(temp, mul8(_mm256_load_ps(&A[(i + 1)*N + k+8]), _mm256_load_ps(&P[j*N + k+8])));
				D[(i+1)*N + j] += temp.m256_f32[0] + temp.m256_f32[1] + temp.m256_f32[2] + temp.m256_f32[3] +
					temp.m256_f32[4] + temp.m256_f32[5] + temp.m256_f32[6] + temp.m256_f32[7];

				temp = set1(0);				
				temp = add8(temp, mul8(_mm256_load_ps(&A[i*N + k]), _mm256_load_ps(&P[(j+1)*N + k])));
				temp = add8(temp, mul8(_mm256_load_ps(&A[i*N + k+8]), _mm256_load_ps(&P[(j + 1)*N + k+8])));
				D[i*N + j+1] += temp.m256_f32[0] + temp.m256_f32[1] + temp.m256_f32[2] + temp.m256_f32[3] +
					temp.m256_f32[4] + temp.m256_f32[5] + temp.m256_f32[6] + temp.m256_f32[7];

				temp = set1(0);
				temp = add8(temp, mul8(_mm256_load_ps(&A[(i + 1)*N + k]), _mm256_load_ps(&P[(j + 1)*N + k])));
				temp = add8(temp, mul8(_mm256_load_ps(&A[(i + 1)*N + k+8]), _mm256_load_ps(&P[(j + 1)*N + k+8])));
				D[(i + 1)*N + j+1] += temp.m256_f32[0] + temp.m256_f32[1] + temp.m256_f32[2] + temp.m256_f32[3] +
					temp.m256_f32[4] + temp.m256_f32[5] + temp.m256_f32[6] + temp.m256_f32[7];
			}
}

void MatrixMult_SIMD2()
{
	std::thread t[nbThreads];
	for (int i = 0; i < nbThreads; i++)
		t[i] = std::thread(setElementsSIMD2, i);
	//Join 
	for (int i = 0; i < nbThreads; i++) {
		t[i].join();
	}
}

// Multithreading - extra check ------------------------------------------------
void setElem(int x)
{
	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i++)
		for (int j = 0; j < N; j++)
			D[i*N + j] = 0;

	for (int i = x*N / nbThreads; i < (x + 1)* N / nbThreads; i++)
		for (int k = 0; k < N; k++)
			for (int j = 0; j < N; j++)
			//for (int k = 0; k < N; k++)
				D[i*N + j] += A[i*N + k] * B[k*N + j];

}

void Mutithreading_ExtraCheck()
{
	std::thread t[nbThreads];
	for (int i = 0; i < nbThreads; i++)
		t[i] = std::thread(setElem, i);
	//Join 
	for (int i = 0; i < nbThreads; i++) {
		t[i].join();
	}
}


// Test ---------------------------------------------------------------
void compareResults()
{
	bool ok = true;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			if (C[i*N + j] != D[i*N + j])
			{
				ok = false;
			}
		}
	if (ok == false)
		printf("Results are not equal\n\n");
	else printf("Results are equal\n\n");

}

void checkMatrixC()
{
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			std::cout << C[i*N + j];
			std::cout << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void checkMatrixD()
{
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			std::cout << D[i*N + j];
			std::cout << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void check_SIMD()
{
	float8 temp, tempI;
	temp = set1(0);
	tempI = set1(1);
	temp = _mm256_load_ps(&C[0*N + 0]);
	_mm256_store_ps(&D[0*N + 0], temp);

		for (int j = 0; j < 8; j++)
		{
			//std::cout << temp.m256_f32[j];
			std::cout << D[0*N + j];
			std::cout << " ";
		}
	std::cout << "\n";
}
// -----------------------------------------------------------------------

int _tmain(int argc, _TCHAR* argv[])
{
	AllocateAndPopulate();

	Chrono c;
	MatrixMult();
	long long t=c.Elapsed_us();
	printf("Time (Normal): %dms\n\n",t/1000);

	Chrono c2;
	MatrixMult_Mutithreading();
	long long t2 = c2.Elapsed_us();
	printf("Time (Multith.): %dms\n", t2 / 1000);
	compareResults();

	Chrono c6;
	Mutithreading_ExtraCheck();
	long long t6 = c6.Elapsed_us();
	printf("Time (Multith. extra check): %dms\n", t6 / 1000);
	compareResults();
	transpose_Matrix();
	
	Chrono c3;
	MatrixMult_Transposed();
	long long t3 = c3.Elapsed_us();
	printf("Time (Line access): %dms\n", t3 / 1000);
	compareResults();

	Chrono c4;
	MatrixMult_SIMD1();
	long long t4 = c4.Elapsed_us();
	printf("Time (SIMD): %dms\n\n", t4 / 1000);
	checkMatrixC();
	checkMatrixD();

	Chrono c5;
	MatrixMult_SIMD2();
	long long t5 = c5.Elapsed_us();
	printf("Time (SIMD): %dms\n\n", t5 / 1000);
	checkMatrixC();
	checkMatrixD();

	//check_SIMD();
	
	int x;
	std::cin >> x;

	DeAllocate();
	return 0;
}

