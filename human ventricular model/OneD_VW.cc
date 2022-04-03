#include "TP06.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "omp.h"

using namespace std;

#define VENT

int main(int argc, char *argv[])
{
	double dx = 0.15; 
	double dt = 0.02; 
	int epiCellNum = 0;
	int mCellNum = 0;
	int endoCellNum = 0;

	#ifdef VENT	
	endoCellNum = 25;
	mCellNum = 35;
	epiCellNum = 40;
	#endif

	double ventCoeff = 0.154;  

	// for ventricle
	#ifdef VENT
	int numS1 = 2;
	double BCL = 1000; 
	double stopTime = numS1*BCL; 
	double stimStrength = -52;
	double stimDuration = 3;	
	double s2stimStrength = -104.0;
	double s2stimDuration = 3;
	double stimStart = 0.0; 
	#endif


	// --------start simulation--------	
	// CV calculation stuff
	double cvStartTime = 0;
	double cvEndTime = 0;
	int cvStartFlag = 0; // 0 for not start yet
	int cvEndFlag = 0;
	double cv = 0;

	// parallel stuff
	int coreNum = 8;//omp_get_num_procs();
	omp_set_num_threads(2 * coreNum);


	int cellNum = epiCellNum + mCellNum + endoCellNum; 
	typedef CellType* CellPointer;
	TP06* strand[cellNum]; 
	double coeff[cellNum]; // diffusion parameters for each cell
	double dcoeff_dx[cellNum]; // first order derivative for each cell
	double oldV[cellNum];

	// assign coeff according to cell type
	for(int i = 0; i < cellNum; i++)
	{
		// Step 1: general case without yet considering transition in the conjunction of two heterogenous tissues
		#ifdef VENT // set coeff to 'ventCoeff' whatever it was if VENT defined.
		if (i == 60)
			coeff[i] = 0.2*ventCoeff;
		else
			coeff[i] = ventCoeff;
		#endif
	}

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

		#ifdef VENT
		if(i >= 0 && i < endoCellNum)
		{
			strand[i] = new TP06(ENDO);

		}
		else if (i < endoCellNum + mCellNum)
		{
			strand[i] = new TP06(MCELL);
		}
		else // i < total cellnum
		{
			strand[i] = new TP06(EPI);
		}	
		
		#endif
		strand[i]->setDt(dt);
	}
	
	#ifdef VENT
	FILE *datafile = fopen("Outputs/VentOneDResults_VW.dat","w+");
	#endif

	double time = 0;
	int step = 0;
	FILE* FF;

	double left = 355;  
	double right =355; 
	#define MODE1
	#define MODE2 
	#define MODE3 
	int tempstart =10;

	double boundary = 10; //negative
	double s2startTime = 0.5*(left + right);
	double maxEndo = -100;
	double maxEpi = -100;
	int flag = 0;
	
	while(right - left >= 0.1 || right == left)
	{
		char VoltFileName[200];
		sprintf(VoltFileName, "Transmural1D_s2@%.2f.dat", s2startTime);
		FF = fopen(VoltFileName,"w");
		fclose(FF);
		#pragma omp parallel for schedule(static)
		for(int i = 0; i < cellNum; i++)
		{
			strand[i]->init(strand[i]->getCellType());
		}


		for(time = 0.0, step = 0; time <= stopTime; time += dt, step++)
		{
			if(step%20000 == 0) // 2e4 * 5e-6 = 0.1s 
			{
				cout << "s2startTime = " << s2startTime << "ms, Progress = " << 100.0*time/stopTime << "\%." << endl;
			}


			for(int i = 0; i < cellNum; i++)
			{
				oldV[i] = strand[i]->getV();
			}
			
			#pragma omp parallel for schedule(static)
			for(int i = 0; i < cellNum; i++)
			{
				strand[i]->setIstim(0.0);
				if(time - floor(time/BCL)*BCL >= stimStart && 
			   	   time - floor(time/BCL)*BCL < stimStart + stimDuration)
				{
			    	if(i < 3)
					{
						strand[i]->setIstim(stimStrength);
					}
				}		
				else if(time - (numS1-1)*BCL >= s2startTime && 
			   			time - (numS1-1)*BCL < s2startTime + s2stimDuration) 
				{
					if(i >= tempstart && i < tempstart+3)
					{
						strand[i]->setIstim(s2stimStrength);
					}
				}
				
				// ---------calculate diffusion, i.e. dVgap---------
			
				double dVgap_dt = 0;
				double first_order;
				double second_order;

				// Step 1: calculate first and second order of membrane potential
				if(i == 0) 
				{
					// use strand[0] instead of "strand[-1]"
					first_order = (oldV[i+1] - oldV[i])/(1.0*dx);
					second_order = (oldV[i+1] + oldV[i] - 2.0*oldV[i])/(dx*dx);
				}
				else if(i > 0 && i < cellNum - 1) 
				{
					// normal case
					first_order = (oldV[i+1] - oldV[i-1])/(2.0*dx);
					second_order = (oldV[i+1] + oldV[i-1] - 2.0*oldV[i])/(dx*dx);
				}
				else if(i == cellNum - 1)
				{
					// use oldV[cellNum-1] instead of "oldV[cellNum]" as the latter is out of index
					first_order = (oldV[i] - oldV[i-1])/(1.0*dx);
					second_order = (oldV[i] + oldV[i-1] - 2.0*oldV[i])/(dx*dx);
				}

				// Step 2: calculate dVgap according to equations
				dVgap_dt = dcoeff_dx[i]*first_order + coeff[i]*second_order;
				strand[i]->setDVgap_dt(dVgap_dt);
				strand[i]->update();
			}// end cell loop

			// 3. output file. Unfortunately, this part cannot run in paralell
			if( floor(time/BCL) == numS1 - 1) // output final cycle only
			// if(step%50 == 0) // 50*dt = 1 ms once
			{
				FF = fopen(VoltFileName,"a");
				for(int j = 0; j < cellNum; j++)
				{					
					if(step%50 == 0) // 1ms once
					{				
						if(j==0)
							fprintf(FF,"%4.10f\t",time); 				
						fprintf(FF,"%4.10f\t",strand[j]->getV()); // unit: mV						
						if(j == cellNum-1) // write '\n' at line end
							fprintf(FF,"\n");
					}	
				}
				fclose(FF);
			}// end Membrane Potential recording



			// get maxEpi and maxEndo during s2
			if(time - (numS1-1)*BCL >= s2startTime ) 
			{
				if(strand[cellNum-1]->getV() >= maxEpi) maxEpi = strand[cellNum-1]->getV();
				if(strand[0]->getV() >= maxEndo) maxEndo = strand[0]->getV();
			}


		}// end of timeloop
		// fclose(datafile);
		

		cout << "maxEndo = " << maxEndo << endl;
		cout << "maxEpi = " << maxEpi << endl;
		cout << endl;


		// *************mode 1: divide and find*****************

		#ifdef MODE1
		if(maxEndo+boundary < 0 && maxEpi+boundary < 0)
		{
			left = s2startTime;
			s2startTime = 0.5*(left + right);
		}
		else if (maxEndo+boundary > 0 && maxEpi+boundary > 0)
		{
			right = s2startTime;
			s2startTime = 0.5*(left + right);
		}
		else if (  (maxEndo + boundary) * (maxEpi + boundary) <= 0 )
		{
			flag = 1;
		}

		if (flag == 1)	
			break;
		#endif


		// *************mode 2: find left boundary (left must equal to right and must left to VW)*****************
		#ifdef MODE2
		if (  (maxEndo + boundary) * (maxEpi + boundary) <= 0 )
		{
			flag = 1;
		}
		else
		{
			left += 0.1;//0.00001;
			right += 0.1;//0.00001;
			s2startTime = left;
		}

		if (flag == 1)	
			break;
		#endif


		// *************mode 3: find right boundary (left must equal to right and must right to VW)*****************
		#ifdef MODE3
		if (  (maxEndo + boundary) * (maxEpi + boundary) <= 0 )
		{
			flag = 1;
		}
		else
		{
			left -= 0.1;//0.00001;
			right -= 0.1;//0.00001;
			s2startTime = left;
		}

		if (flag == 1)	
			break;
		#endif

		// **************VW search finished here**************


		// reinitialization
		maxEndo = -100;
		maxEpi = -100;
	}

	if(flag == 1) cout << "VM found!" << endl;
	else cout << "No VM!" << endl;

	printf("All done.\n");
	return 0;
}