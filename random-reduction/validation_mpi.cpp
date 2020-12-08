#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
 
#define epsilon 1.e-8

using namespace std;

int main(int argc, char* argv[]){

	double **Acpu, **Bcpu, **Gcpu, **Ampi, **Bmpi, **Gmpi;

	double rAcpu, rBcpu, rGcpu, rAmpi, rBmpi, rGmpi, resA,resB,resG;
	int N,M;

	string P; 


	if(argc>1){
		P = argv[1];
	}
	ifstream alphasCPU("Alphas.mat");
	if(!(alphasCPU.is_open())){
		cout<<"Error: Alphas.mat file not found"<<endl;
		return 0;
	}

	ifstream alphasMPI("AlphasMPI.mat");
	if(!(alphasMPI.is_open())){
		cout<<"Error: AlphasMPI.mat file not found"<<endl;
		return 0;
	}

	ifstream betasCPU("Betas.mat");
	if(!(betasCPU.is_open())){
		cout<<"Error: Betas.mat file not found"<<endl;
		return 0;
	}
	ifstream betasMPI("BetasMPI.mat");
	if(!(betasMPI.is_open())){
		cout<<"Error: Gammas.mat file not found"<<endl;
		return 0;
	}

	ifstream gammasCPU("Gammas.mat");
	if(!(gammasCPU.is_open())){
		cout<<"Error: Gammas.mat file not found"<<endl;
		return 0;
	}

	ifstream gammasMPI("GammasMPI.mat");
	if(!(gammasMPI.is_open())){
		cout<<"Error: GammasMPI.mat file not found"<<endl;
		return 0;
	}

	alphasCPU>>M;
	alphasCPU>>N;


	Acpu = new double *[N];
	Bcpu = new double *[N];
	Gcpu = new double *[N];

	Ampi = new double *[N];
	Bmpi = new double *[N];
	Gmpi = new double *[N];

	for(int i =0;i<N;i++){
		Acpu[i] = new double[N];
		Bcpu[i] = new double[N];
		Gcpu[i] = new double[N];

		Ampi[i] = new double[N];
		Bmpi[i] = new double[N];
		Gmpi[i] = new double[N];
	}

	for(int i =0; i<M;i++){
		for(int j =0; j<N;j++){

			alphasCPU>>Acpu[i][j];
			
			betasCPU>>Bcpu[i][j];
			gammasCPU>>Gcpu[i][j];

			alphasMPI>>Ampi[i][j];
			betasMPI>>Bmpi[i][j];
			gammasMPI>>Gmpi[i][j];
		}
	}

	alphasCPU.close();
	betasCPU.close();
	gammasCPU.close();

	alphasMPI.close();
	betasMPI.close();
	gammasMPI.close();




	rAcpu = 0.0;
	rBcpu = 0.0;
	rGcpu =0.0;
	rAmpi = 0.0;
	rBmpi = 0.0;
	rGmpi = 0.0;

	for(int i = 0; i<N; i++){
		for(int j =0; j<N;j++){

			rAcpu += (abs(Acpu[i][j]));
			rBcpu += (abs(Bcpu[i][j]));
			rGcpu += (abs(Gcpu[i][j]));

			rAmpi += (abs(Ampi[i][j]));
			rBmpi += (abs(Bmpi[i][j]));
			rGmpi += (abs(Gmpi[i][j]));
		}
	}
	resA = rAcpu - rAmpi;
	resB = rBcpu - rBmpi;
	resG = rGcpu - rGmpi;

	if((abs(resA)<=epsilon) && (abs(resB) <= epsilon) && (resG <= epsilon)){
		cout<<"VALID!"<<endl<<endl;
	}

	else{
		cout<<"NOT VALID!"<<endl<<endl;
	}

	if(P == "-p"){

		cout<<"difference in Alphas: "<<resA<<endl<<"difference in Betas: "<<resB<<endl<<"difference in Gammas: "<<resG<<endl;
	}

	return 0;
}
