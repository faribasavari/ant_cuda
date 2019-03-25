#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>
#include<curand_kernel.h>
#include<curand.h>
#define MAX_TIME 1
#define MAX_ANTS 1
#define Q 100
#define ALPHA 2                  //we used alpha 10 beta 1 for u_c_lohi
#define BETA 1
#define RHO 1.5 
#define ntask 512
#define nres 16
#define evaporation 0.5

using namespace std;

struct ant {
	int curJob, nextJob;
	int visited[ntask];
	int solution[MAX_TIME][ntask];
	float makespan[MAX_TIME][nres];
};
struct job {

	double res[nres];
};

int NC = 0;

job jobs[ntask];
double sum, F;
float make[nres];
ant ants[MAX_ANTS];
double freeRes[nres];
double pheromone[ntask][nres], Delta[ntask][nres], heuristic[ntask][nres], probability[ntask][nres],valid[ntask][nres];
curandState  state[MAX_ANTS];


__global__ void setup_curand_states(curandState *state_d, int t) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(t, id, 0, &state_d[id]);
}
__device__ float generate(curandState* globalState, int ind) {
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}
__global__ void initialize(float *d_pheromone, float *d_delta, float *d_heuristic, job *d_job, int task, int res, int s, float max)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;        //res
	int row = blockIdx.y * blockDim.y + threadIdx.y;      //task
	if ((row<task) && (col<res)) {
		d_heuristic[col + row * res] = 1 / max;
		d_delta[col + row * res] = (1 - evaporation) / max;
		d_pheromone[col + row * res] = 0;
		d_pheromone[col + row * res] = evaporation*d_pheromone[col + row * res] + d_delta[col + row * res];
	}

}
__device__ int findmax_probability(float *d_probability, ant *d_ant, job *d_job, float *d_free, int k,int nc) {
	int i, j = 0, maxj = 0, maxi = 0;
	double max = d_probability[0];
	for (i = 0; i < ntask; i++) {
		if (d_ant[k].visited[i] == 0)
			for (j = 0; j < nres; j++) {
				if (d_probability[j + nres*i] > max /*&& d_ant[k].tabu[i]==0*/) {
					max = d_probability[j + nres*i];
					maxi = i;
					maxj = j;
				}
			}
	}
	d_ant[k].solution[nc][maxi] = maxj;
	d_ant[k].makespan[nc][maxj] += d_job[maxi].res[maxj];
	for (int g = 0; g < nres; g++) {
		d_probability[g + maxi*nres] = 0;
		d_free[g] = d_free[g] - d_job[maxi].res[g];

	}
	return maxi;
}
__device__ double findmax(float *a) {
	float max = a[0];
	for (int i = 1; i < nres; i++) {
		if (a[i] > max) {
			max = a[i];
		}
	}
	return max;
}

__device__ int selectNextJob(float *d_probability, float *d_pheromone, float *d_delta, float *d_heuristic, job *d_job, int k, int n, float *d_free, ant *d_ant, float sum, int nc, curandState *state_d)
{
	int i;/// = ants[k].curJob;
	int j, nextJob;
	float max;

	sum = 0;
	max = findmax(d_free);
	for (i = 0; i < ntask; i++) {

		for (j = 0; j < nres; j++) {
			d_heuristic[j + nres*i] = 1 / d_free[j];
			d_delta[j + nres*i] += (1 - evaporation) / max;
			d_pheromone[j + nres*i] = evaporation * d_pheromone[j + nres*i] + d_delta[j + nres*i];
			sum += powf(d_pheromone[j + nres*i], ALPHA) * powf(d_heuristic[j + nres*i], BETA) * 1 / d_job[i].res[j];
		}
	}

	for (i = 0; i < ntask; i++) {            //calculate probability for any task and any resource
		if (d_ant[k].visited[i] == 0) {
			for (j = 0; j < nres; j++) {
				d_probability[j + nres*i] = (powf(d_pheromone[j + nres*i], ALPHA) * powf(d_heuristic[j + nres*i], BETA) * 1 / d_job[i].res[j]) / sum;
			}
		}
	}
	nextJob = findmax_probability(d_probability, d_ant, d_job, d_free, k, nc);
	
	return nextJob;
}


__global__ void select(int nc, float *d_pheromone, float *d_delta, float *d_heuristic, job *d_job, ant *d_ant, float *d_probability, float *d_free, int n, float sum, curandState *state_d)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < MAX_ANTS) {
		for (int s = 1; s<n; s++)
		{
			int j = selectNextJob(d_probability, d_pheromone, d_delta, d_heuristic, d_job, id, n, d_free, d_ant, sum, nc, state_d);
			//printf("j:%d\n", j);
			d_ant[id].nextJob = j;
			d_ant[id].visited[j] = 1;
			d_ant[id].curJob = j;
		}
	}
	
	
}
__global__ void firststep(ant *d_ant, job *d_job,int nc) {
	int randres, randtask;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id<MAX_ANTS) {

		//randtask = (blockIdx.x + clock() + clock() * threadIdx.x *blockDim.x) % ntask;
		// randres=( blockIdx.x + clock() + clock()*threadIdx.x * blockDim.x)%nres;
		curandState state;
		curand_init((unsigned long long)clock(), id, 0, &state);
		double rand1 = curand_uniform_double(&state);
		double rand2 = curand_uniform_double(&state);
		randtask = (int)((rand1 / rand2)*blockIdx.x) % ntask;
		randres = (int)((rand1 / rand2)*id) % nres;
	//	printf("rand1:%d , rand2:%d \n", randres, randtask);
		d_ant[id].curJob = randtask;
		for (int i = 0; i<ntask; i++)
		{
			d_ant[id].visited[i] = 0;
		}
		
		d_ant[id].visited[randtask] = 1;
		for (int i = 0; i<nres; i++)
			d_ant[id].makespan[nc][i] = 0;
		d_ant[id].makespan[nc][randres] = d_job[randtask].res[randres];
		d_ant[id].solution[nc][randtask] = randres;
	}
}
__global__ void emptyTabu(ant *d_ant, float *d_delta, int n) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		for (int s = 0; s<n; s++) {

			d_ant[id].visited[s] = 0;
		}
	}
}
__global__ void updatePheromone(float *d_pheromone, float *d_delta, float *d_heuristic, job *d_job, int n, int max, float s) {



	s = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;        //res
	int row = blockIdx.y * blockDim.y + threadIdx.y;      //task


	if ((row<ntask) && (col<nres)) {

		d_heuristic[col + row * nres] = 1 / max;
		d_delta[col + row * nres] += (1 - evaporation) / max;
		d_pheromone[col + row * nres] = evaporation*d_pheromone[col + row * nres] + d_delta[col + row * nres];

	}

}
int main(int argc, char *argv[])
{
	if (argc > 1) {
		cout << "Reading File " << argv[1] << endl;
	}
	else {
		cout << "Usage:progname inputFileName" << endl;
		return 1;
	}
	int i, j, antindex, spanindex;
	double max;
	ifstream in;
	
	in.open(argv[1]);
	for (i = 0; i<nres; i++)
	{
		for (j = 0; j<ntask; j++) {

			in >> jobs[j].res[i];
			//cout<<jobs[j].res[i]<<"\t";	
			freeRes[i] = freeRes[i] + jobs[j].res[i];
		}
		cout << endl;
	}
	max = freeRes[0];
	for (i = 1; i < nres; i++) {
		if (freeRes[i] > max) {
			max = freeRes[i];
		}
	}
	clock_t begin = clock();
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((ntask - 1) / 16 + 1, (ntask - 1) / 16 + 1, 1);
	float *d_pheromone, *d_delta, *d_free, *d_heuristic, *d_probability;
	ant *d_ant;
	job *d_job;
	curandState  *state_d;
	cudaMalloc((void**)&d_pheromone, sizeof(float) * ntask * nres);
	cudaMalloc((void**)&d_free, sizeof(float) * nres);
	cudaMalloc((void**)&d_delta, sizeof(float) * ntask * nres);
	cudaMalloc((void**)&d_ant, sizeof(ant) * MAX_ANTS);
	cudaMalloc((void**)&d_job, sizeof(job) * ntask);
	cudaMalloc((void**)&d_heuristic, sizeof(float) * ntask *nres);
	cudaMalloc((void**)&state_d, sizeof(state));
	cudaMalloc((void**)&d_probability, sizeof(float) * ntask * nres);
	cudaMemcpy(d_job, jobs, sizeof(job) * ntask, cudaMemcpyHostToDevice);
	srand(time(0));
	int seed = rand();
	setup_curand_states << < (ntask - 1) / 32 + 1, 16 >> > (state_d, seed);
	initialize << <gridDim, blockDim >> >(d_pheromone, d_delta, d_heuristic, d_job, ntask, nres, sum, max);
	cudaThreadSynchronize();
	for (;;)
	{	//	cout<<randres<<","<<randtask<<endl;
		firststep << <(ntask - 1) / 32 + 1, 16 >> >(d_ant, d_job, NC);	
		cudaThreadSynchronize();
		select << <(ntask - 1) / 32 + 1, 16 >> >(NC, d_pheromone, d_delta, d_heuristic, d_job, d_ant, d_probability, d_free, ntask, sum,state_d);
        cudaThreadSynchronize();
		cudaMemcpy(freeRes, d_free, sizeof(float) * nres, cudaMemcpyDeviceToHost);
		max = freeRes[0];
		for (int i = 1; i < nres; i++) {
			if (freeRes[i] > max) {
				max = freeRes[i];
			}
		}
		cudaMemcpy(d_free, freeRes, sizeof(float) * nres, cudaMemcpyHostToDevice);
		cudaMemcpy(ants, d_ant, sizeof(ant) * MAX_ANTS, cudaMemcpyDeviceToHost);
		cudaMemcpy(jobs, d_job, sizeof(job) * ntask, cudaMemcpyDeviceToHost);
		cudaMemcpy(freeRes, d_free, sizeof(float) * nres, cudaMemcpyDeviceToHost);
		updatePheromone << < (ntask - 1) / 32 + 1, 32 >> >(d_pheromone, d_delta, d_heuristic, d_job, ntask, max, sum);
		cudaThreadSynchronize();
		NC += 1;
		if (NC < MAX_TIME) {
			emptyTabu << <(ntask - 1) / 32 + 1, 16 >> >(d_ant, d_delta, ntask);
			cudaMemcpy(ants, d_ant, sizeof(ant) * MAX_ANTS, cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
		}
		else {
			break;
		}
	}//end of for(;;)
	
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	
	/*******************************************************************/

	cudaFree(d_free);
	cudaFree(d_pheromone);
	cudaFree(d_delta);
	cudaFree(d_heuristic);
	cudaFree(d_probability);
	cudaFree(d_job);
	cudaFree(d_ant);
	
	cout<<"_________________\n";
	for (int i = 0; i<MAX_ANTS; i++) {
		max = 0;
		for (int y = 0; y<MAX_TIME; y++) {
			max = ants[i].makespan[y][0];
			antindex = i;
			spanindex = y;
			for (int q = 1; q < nres; q++) {
				printf("span:%f\n",ants[i].makespan[y][q]);
				if (ants[i].makespan[y][q] > max) {
					max = ants[i].makespan[y][q];
					antindex = i;
					spanindex = y;
				}
			}
			cout << "max makespan" << max << "\t";
			cout << endl;

		}
	}
	cout << "time:" << time_spent;



	return 0;
}
