#include "TerkRatLV.h"
using namespace std;


int main(int argc, char *argv[])
{
	double BCL = 1000; 
	double numS1 = 1000;
	double dt = 0.005; 
	double stopTime = numS1*BCL; 
	double stimStrength = -6.0;
	double stimDuration = 5;
	double stimStart = 25.0; 
	typedef TerkRatLV CellType;
	typedef TerkRatLV* CellPointer;

	double apd20;
	double apd25;
	double apd50;
	double apd75;
	double apd80;
	double apd90;
	double dvdt_max;
	double rest_potential;
	double amplitude;
	double overshoot;

	CellPointer cell = new CellType(LVEPI);

	FILE *datafile = fopen("Outputs/singlecell_results.dat","w+");
	FILE *apdfile = fopen("Outputs/singlecell_apd.dat","w+");
	/**
	FILE *statefile = fopen("Outputs/SingleCell_data/state_epi.dat","r");
	cell->readinAllStates(statefile);
	fclose(statefile);
	**/
	cell->setDt(dt);
	double time = 0;
	double t_maxdvdt;
	int step = 0;
	double oldV, oldDvdt, repo20, repo25, repo50, repo75, repo80, repo90;
	bool peakfound;
	for(time = 0.0, step = 0; time < stopTime; time += dt, step++)
	{
		if(time - floor(time/BCL)*BCL >= stimStart && 
		   time - floor(time/BCL)*BCL < stimStart + stimDuration)	
		{
			cell->setIstim(stimStrength);
		}
		else
		{
			cell->setIstim(0.0);
		}

	
		 //2. update all states and currents
		cell->update();
		
		//if(step%(int(1/dt)) == 0) // 5*dt = 1ms once
		if (floor(time/BCL)>=numS1-10)
		{
			if(step%(int(5/dt)) == 0){
				fprintf(datafile,"%4.10f\t", time-(numS1-10)*BCL); 
				fprintf(datafile,"%4.10f\t", cell->getV()); 
				fprintf(datafile,"%4.10f\t", cell->getCai()); 
				fprintf(datafile,"%4.10f\t", cell->getCaSR()); 
				fprintf(datafile,"%4.10f\t", cell->getIt());  
				fprintf(datafile,"%4.10f", cell->getICaL());  
				fprintf(datafile,"\n");
			}
		}
	
		if (floor(time/BCL)==numS1-2  && floor((time+dt)/BCL)==numS1-1)
		{
			//cell->outputAllStates(statefile);
			rest_potential = cell->getV();
			oldDvdt = cell->getAbsDvdt();
			oldV = cell->getV();
			peakfound = false; 
		}

		if (floor((time-stimStart-stimDuration)/BCL) == numS1-1 )
		{
			if(cell->getV() > oldV)
			{  
				oldV = cell->getV();
				if( cell->getAbsDvdt() > dvdt_max )
				{
					dvdt_max = cell->getAbsDvdt();
				}
			}	
			else if(cell->getV() <= oldV && !peakfound) // peak not found yet
			{
				peakfound = true;
				overshoot = oldV;
				amplitude = overshoot - rest_potential; // should always be a positive value
				repo20 = overshoot - 0.20*amplitude;
				repo25 = overshoot - 0.25*amplitude;
				repo50 = overshoot - 0.50*amplitude;
				repo75 = overshoot - 0.75*amplitude;
				repo80 = overshoot - 0.80*amplitude;
				repo90 = overshoot - 0.90*amplitude;
				oldV = cell->getV();

			}
			else if(cell->getV() <= oldV && peakfound) // peak already found
			{
				if(oldV >= repo20 && cell->getV() <= repo20)
					apd20 = time - floor(time/BCL)*BCL - stimStart - stimDuration; // note that apd calculate from the stimStart.
				else if(oldV >= repo25 && cell->getV() <= repo25)
					apd25 = time - floor(time/BCL)*BCL - stimStart - stimDuration; // note that apd calculate from the stimStart.
				else if(oldV >= repo50 && cell->getV() <= repo50)
					apd50 = time - floor(time/BCL)*BCL - stimStart - stimDuration; // note that apd calculate from the stimStart.
				else if(oldV >= repo75 && cell->getV() <= repo75)
					apd75 = time - floor(time/BCL)*BCL - stimStart - stimDuration; // note that apd calculate from the stimStart.
				else if(oldV >= repo80 && cell->getV() <= repo80)
					apd80 = time - floor(time/BCL)*BCL - stimStart - stimDuration; // note that apd calculate from the stimStart.
				else if(oldV >= repo90 && cell->getV() <= repo90)
					apd90 = time - floor(time/BCL)*BCL - stimStart - stimDuration; // note that apd calculate from the stimStart.
				oldV = cell->getV();
			}
		}

		if (time + dt >= stopTime)
		{	// APD20, APD50, APD90, ,  , , 
			fprintf(apdfile,"Resting membrane potential = %.5f mV\n",rest_potential); // unit: mV
			fprintf(apdfile,"Maximum dV/dt (UpstrokeVelocity_max) = %.5f mV/ms\n",dvdt_max); // unit: mV/ms
			fprintf(apdfile,"Overshoot = %.5f mV\n",overshoot); // unit: mV	
			fprintf(apdfile,"Amplitude = %.5f mV\n",amplitude); // unit: mV
			fprintf(apdfile,"APD20 = %.5f ms\n",apd20); // unit: ms		
			fprintf(apdfile,"APD25 = %.5f ms\n",apd25); // unit: ms		
			fprintf(apdfile,"APD50 = %.5f ms\n",apd50); // unit: ms	
			fprintf(apdfile,"APD75 = %.5f ms\n",apd75); // unit: ms		
			fprintf(apdfile,"APD80 = %.5f ms\n",apd80); // unit: ms		
			fprintf(apdfile,"APD90 = %.5f ms\n",apd90); // unit: ms
		}

		// 6. output apd statistics to screen
		if (time + dt >= stopTime)
		{	// APD20, APD50, APD90, ,  , , 
			printf("Resting membrane potential = %.5f mV\n",rest_potential); // unit: mV
			printf("Maximum dV/dt (UpstrokeVelocity_max) = %.5f mV/ms\n",dvdt_max); // unit: mV/ms
			printf("Overshoot = %.5f mV\n",overshoot); // unit: mV	
			printf("Amplitude = %.5f mV\n",amplitude); // unit: mV
			printf("APD20 = %.5f ms\n",apd20); // unit: ms		
			printf("APD25 = %.5f ms\n",apd25); // unit: ms		
			printf("APD50 = %.5f ms\n",apd50); // unit: ms	
			printf("APD75 = %.5f ms\n",apd75); // unit: ms		
			printf("APD80 = %.5f ms\n",apd80); // unit: ms		
			printf("APD90 = %.5f ms\n",apd90); // unit: ms
		}
	}
	fclose(datafile);
	fclose(apdfile);
	return 0;
}




