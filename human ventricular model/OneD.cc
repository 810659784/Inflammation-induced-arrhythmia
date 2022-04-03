#include "TP06.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "omp.h"

using namespace std;


int main(int argc, char *argv[])
{
	double dx = 0.15; // mm
	double dt = 0.02; // ms
	int endoCellNum = 25;
	int MCellNum = 35;
	int epiCellNum = 40;
	double humanCoeff = 0.154;


	int numS1 = 5;
	double BCL = 1000; // ms   325(2:1)   250(1:1)  1000(EAD) 500(normal) 750(normal)
	double stopTime = numS1*BCL; //ms
	double stimStrength = -80.0;//8.78; //8.78;//-8.78; // pA
	double stimDuration = 1;	// ms
	double stimStart = 25; // ms  // indicates the time point of beginning stimulus in a cycle


	double cvStartTime = 0;
	double cvEndTime = 0;
	int cvStartFlag = 0; // 0 for not start yet
	int cvEndFlag = 0;
	double cv = 0;

	// parallel stuff
	int coreNum = 8;//omp_get_num_procs();
	omp_set_num_threads(2 * coreNum);

	// strand initilization, diffusion stuff
	int cellNum = epiCellNum + MCellNum + endoCellNum; // number of cells in OneD strand
	typedef Cell* CellPointer;
	TP06* strand[cellNum]; // note that constructor contains the initializer
	//ORdHumanVentricle* strand[cellNum];
	double coeff[cellNum]; // diffusion parameters for each cell
	double dcoeff_dx[cellNum]; // first order derivative for each cell
	double oldV[cellNum];

	// assign coeff according to cell type
	for(int i = 0; i < cellNum; i++)
		coeff[i] = humanCoeff;


	// Calculate the dcoeff/dx(i.e. dcoeff_dx in the code) in the 1D strand
	for(int i = 0; i < cellNum; i++)
	{
		if (i == 0) 
			dcoeff_dx[i] = (coeff[i+1] - coeff[i])/dx;
		else if (i == cellNum-1) 
			dcoeff_dx[i] = (coeff[i] - coeff[i-1])/dx;
		else
			dcoeff_dx[i] = (coeff[i+1] - coeff[i-1])/(2.0*dx);
	}

	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < cellNum; i++)
	{

		if(i >= 0 && i < endoCellNum)
		{
			strand[i] = new TP06(ENDO);
		}else if(i>=endoCellNum&&i<endoCellNum+MCellNum)
		{
			strand[i] = new TP06(MCELL);
		}
		else // i < total cellnum
		{
			strand[i] = new TP06(EPI);
		}	
		strand[i]->setDt(dt);
	}

	FILE *datafile = fopen("Outputs/TP06OneDResults_test.dat","w+");


	double time = 0;
	int step = 0;
	for(time = 0.0, step = 0; time <= stopTime; time += dt, step++)
	{
		if(step%10000 == 0) // 1000 * dt ms = 50 ms 
			cout << "Progress = " << 100.0*time/stopTime << "\%." << endl;

		for(int i = 0; i < cellNum; i++)
		{
			oldV[i] = strand[i]->getV();
		}
		for(int i = 0; i < cellNum; i++)
		{
			strand[i]->setIstim(0.0);
			if(time - floor(time/BCL)*BCL >= stimStart && 
		   	   time - floor(time/BCL)*BCL < stimStart + stimDuration)
			{
		    	if(i < 3 && i >= 0)
				{// cells get stimulation in certain duration
					strand[i]->setIstim(stimStrength);
				}
			}
		
			double dVgap_dt = 0;
			double first_order;
			double second_order;
			if(i == 0) 
			{
				first_order = (oldV[i+1] - oldV[i])/(1.0*dx);
				second_order = (oldV[i+1] + oldV[i] - 2.0*oldV[i])/(dx*dx);
			}
			else if(i > 0 && i < cellNum - 1) 
			{
				first_order = (oldV[i+1] - oldV[i-1])/(2.0*dx);
				second_order = (oldV[i+1] + oldV[i-1] - 2.0*oldV[i])/(dx*dx);	
			}
			else if(i == cellNum - 1)
			{
				first_order = (oldV[i] - oldV[i-1])/(1.0*dx);
				second_order = (oldV[i] + oldV[i-1] - 2.0*oldV[i])/(dx*dx);	
			}

			dVgap_dt = dcoeff_dx[i]*first_order + coeff[i]*second_order;
			strand[i]->setDVgap_dt(dVgap_dt);
			strand[i]->update();
			/**

			if (floor(time/BCL)==numS1-2  && floor((time+dt)/BCL)==numS1-1)
			{
				char tempfile[30];
				sprintf(tempfile,"%d.dat",i);
				FILE *statefile=fopen(tempfile,"w+");
				strand[i]->outputAllStates(statefile);
				fclose(statefile);

			}
			**/
		}// end cell loop

		if(step%10 == 0) 
		{
			for(int j = 0; j < cellNum; j++)
			{
				if(j == 0)
					fprintf(datafile,"%4.10f\t", time);
				fprintf(datafile,"%4.10f\t", strand[j]->getV()); // unit: mV
				if(j == cellNum - 1)
					fprintf(datafile,"\n");
			}
		}
		
		if (floor(time/BCL) == numS1-1)
		{ 
			if(strand[5]->getV()>= -30 && cvStartFlag == 0)
			{
				cvStartTime = time;
				cout << "start = " << cvStartTime << endl;
				cvStartFlag = 1;
			}
			if(strand[25]->getV() >= -30 && cvEndFlag == 0)
			{
				cvEndTime = time;
				cout << "end = " << cvEndTime << endl;
				cvEndFlag = 1;
				cv = (dx * 20) / (cvEndTime - cvStartTime);
				cout << "duration = " << cvEndTime - cvStartTime << endl;
			}
		}

	}
	fclose(datafile);


	if(cvStartFlag == 1 && cvEndFlag == 1)
		cout << "CV = " << cv << " m/s." << endl;
	else
		cout << "Conduction failure!" << endl;
	printf("All done.\n");

	return 0;
}