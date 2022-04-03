#include <iostream>
#include <fstream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <unistd.h> 
#include <sys/stat.h> 
#include <sys/types.h> 

//#define part_Inflammation
//#define all_Inflammation
#include <iomanip>

#include <cuda_runtime.h>

using namespace std;

__constant__ double Ko = 5.4;
__constant__ double Cao = 2.0;
__constant__ double Nao = 140.0;
__constant__ double Vc = 0.016404;
__constant__ double Vsr = 0.001094;
__constant__ double Vss = 0.00005468;
__constant__ double Bufc = 0.2;
__constant__ double Kbufc = 0.001;
__constant__ double Bufsr = 10.;
__constant__ double Kbufsr = 0.3;
__constant__ double Bufss = 0.4;
__constant__ double Kbufss = 0.00025;
__constant__ double Vmaxup = 0.006375;
__constant__ double Kup = 0.00025;
__constant__ double Vrel = 0.102;
__constant__ double k1_ = 0.15;
__constant__ double k2_ = 0.045;
__constant__ double k3 = 0.060;
__constant__ double k4 = 0.005;
__constant__ double EC = 1.5;
__constant__ double maxsr = 2.5;
__constant__ double minsr = 1.;
__constant__ double Vleak = 0.00036;
__constant__ double Vxfer = 0.0038;
__constant__ double R = 8314.472;
__constant__ double F = 96485.3415;
__constant__ double T = 310.0;
      
__constant__ double CAPACITANCE = 0.185;
__constant__ double pKNa = 0.03;
__constant__ double GK1 = 5.405;
__constant__ double GNa = 14.838;
__constant__ double GbNa = 0.00029;
__constant__ double KmK = 1.0;
__constant__ double KmNa = 40.0;
__constant__ double knak = 2.724;
__constant__ double GCaL = 0.00003980;
__constant__ double GbCa = 0.000592;
__constant__ double knaca = 1000;
__constant__ double KmNai = 87.5;
__constant__ double KmCa = 1.38;
__constant__ double ksat = 0.1;
__constant__ double n = 0.35;
__constant__ double GpCa = 0.1238;  
__constant__ double KpCa = 0.0005;
__constant__ double GpK = 0.0146; 

const int X = 600;
const int Y = 600;

const int DimX = Y;
const int ElementNum = X*Y;

const double D1 = 0.154;
const double D2 = 0.154;

//const double D1 = 0.154*0.65;
//const double D2 = 0.154*0.65;

const double DD = D1 - D2;

const double dx = 0.15;
const double dt = 0.02;
const int N1 = 20;

const int Istim   =  0;
const int Cai     =  1;
const int CaSR    =  2;
const int CaSS    =  3;
const int Nai     =  4;
const int Ki      =  5;
const int sm      =  6;
const int sh      =  7;
const int sj      =  8;
const int sxr1    =  9;
const int sxr2    =  10;
const int sxs     =  11;
const int sr      =  12;
const int ss      =  13;
const int sd      =  14;
const int sf      =  15;
const int sf2     =  16;
const int sfcass  =  17;                               
const int sRR     =  18;
const int sOO     =  19;



int *g;
double *V_data;
int curCount = 0;

const int blockSize = 128;

static void HandleError(cudaError_t err, const char *file, int line) {    
	if (err != cudaSuccess){
		cout << cudaGetErrorString(err) << " in " << file << " at line " << line << endl;
		char ch;
		cin >> ch;
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__)) 

void writeData()
{
	int file_id = curCount / 250;         
	std::ostringstream os;      
	os << "ap_" << file_id << ".vtk";
	std::ofstream out(os.str().c_str(), std::ios_base::out);

	out << "# vtk DataFile Version 3.0" << std::endl;
	out << "vtk output" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET STRUCTURED_POINTS" << std::endl;
	out << "DIMENSIONS " << X  << " " << Y  << " " << 1  << std::endl; //X=690 Y=489
	out << "SPACING 1 1 1" << std::endl;
	out << "ORIGIN 0 0 0" << std::endl;
	out << "POINT_DATA " << (X)*(Y)*(1) << std::endl;
	out << "SCALARS ImageFile float 1" << std::endl;
	out << "LOOKUP_TABLE default" << std::endl;

	for (int y = 0; y<Y; y += 1 )
	{
		for (int x = 0; x<X; x += 1 )
		{
			int this_g = g[x*DimX + y];
			if (this_g > 0)
			{
				out << V_data[this_g - 1] << " ";
			}
			else
				out << "-100 ";
		}
		out << std::endl;
	}
	out.close();
}

void writeData_ecg(double time)
{
	std::ostringstream os;
	os << "TP06_2d_data_inflammation";
	std::ofstream out(os.str().c_str(), std::ios_base::out|std::ios_base::app);

	//out << time << " ";
	for (int y = 0; y<Y; y += 1 )
	{
		for (int x = 0; x<X; x += 1 )
		{
			int this_g = g[x*DimX + y];
			if (this_g > 0)
			{
				out << V_data[this_g - 1] << " ";
			}
			else
				out << "-100 ";
		}
		out << std::endl;
	}
	out << std::endl;
	out.close();
}

__global__ void init_u_v(int num, double *u_v)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < num)
	{
		u_v[id]=-86.2;
	}
}

__device__ double valid_g(int _g)
{

	if (_g > 0)
	{
		return _g;
	}

	else if (_g == 0)
	{
		return 0;
	}

	else
	{
		return -_g;
	}
}




__device__ double get_u_v(int host_id, int candidate_id, double *u_v, int *g)
{
	int _g = g[host_id];
	int temp;
	if (_g > 0) 
	{
		return u_v[_g - 1];
	}		
	else if (_g < 0) 
	{
		return u_v[-_g - 1];

	}
	else 
	{
		return -90;  
	}


}

double get_u_v_host(int host_id, int candidate_id, double *u_v, int *g)
{
	int _g = g[host_id];
	int temp;
	if (_g > 0)
	{
		return u_v[_g - 1];
	}

	else if (_g < 0)
	{
		return u_v[-_g - 1]; 
	}

	else
	{
		return -90;
	}
	
}


__global__ void calc_du(int totalNum, double *d, double *u_v, short *typeArray,double *xx, double *yy, double *du, int *map, int *g, int *dev_rev_g)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < totalNum)
	{
		int host_id = dev_rev_g[id+1];

		double this_x = xx[id];
		double this_y = yy[id];

		short type = typeArray[id]; 
		
		int present_x = host_id/DimX;
		int present_y = host_id%DimX;
		
		double df1 = 0;
		double df2 = 0;
		double df3 = 0;
		double df4 = 0;

		double dudxdy = (get_u_v(host_id + DimX + 1, host_id, u_v, g) + get_u_v(host_id - DimX - 1, host_id, u_v, g) - get_u_v(host_id - DimX + 1, host_id, u_v, g) - get_u_v(host_id + DimX - 1, host_id, u_v, g)) / (4 * dx*dx);
		double dudx2 = (get_u_v(host_id - DimX, host_id, u_v, g) + get_u_v(host_id + DimX, host_id, u_v, g) - 2 * get_u_v(host_id, host_id, u_v, g)) / (dx*dx);
		double dudy2 = (get_u_v(host_id - 1, host_id, u_v, g) + get_u_v(host_id + 1, host_id, u_v, g) - 2 * get_u_v(host_id, host_id, u_v, g)) / (dx*dx);
		double dudx = (get_u_v(host_id + DimX, host_id, u_v, g) - get_u_v(host_id - DimX, host_id, u_v, g)) / (2 * dx);
		double dudy = (get_u_v(host_id + 1, host_id, u_v, g) - get_u_v(host_id - 1, host_id, u_v, g)) / (2 * dx);
		int g1 = 0;
		int g2 = 0;

		g1 = g[host_id + DimX];
		g2 = g[host_id - DimX];

		if (g1 > 0 && g2 > 0) df1 = (d[5*(g1 - 1)+1] - d[5*(g2 - 1)+1]) / (2 * dx);
		else if (g1 > 0) df1 = (d[5*(g1 - 1)+1] - d[5*id+1]) / (dx);
		else if (g2 > 0) df1 = (d[5*id+1] - d[5*(g2 - 1)+1]) / (dx);
		else df1 = 0;

		if (g1 > 0 && g2 > 0) df2 = (d[5*(g1 - 1)+2] - d[5*(g2 - 1)+2]) / (2 * dx);
		else if (g1 > 0) df2 = (d[5*(g1 - 1)+2] - d[5*id+2]) / (dx);
		else if (g2 > 0) df2 = (d[5*id+2] - d[5*(g2 - 1)+2]) / (dx);
		else df2 = 0;


		g1 = g[host_id + 1];
		g2 = g[host_id - 1];

		if (g1 > 0 && g2 > 0) df3 = (d[5*(g1 - 1)+3] - d[5*(g2 - 1)+3]) / (2 * dx);
		else if (g1 > 0) df3 = (d[5*(g1 - 1)+3] - d[5*id+3]) / (dx);
		else if (g2 > 0) df3 = (d[5*id+3] - d[5*(g2 - 1)+3]) / (dx);
		else df3 = 0;

		if (g1 > 0 && g2 > 0) df4 = (d[5*(g1 - 1)+4] - d[5*(g2 - 1)+4]) / (2 * dx);
		else if (g1 > 0) df4 = (d[5*(g1 - 1)+4] - d[5*id+4]) / (dx);
		else if (g2 > 0) df4 = (d[5*id+4] - d[5*(g2 - 1)+4]) / (dx);
		else df4 = 0;


		du[id] = ((D2 + (DD*this_x*this_x))*dudx2) + ((D2 + (DD*this_y*this_y))*dudy2) + 
			2 * (DD*this_x*this_y*dudxdy) +
			df1*dudx + df2*dudy + 
			df3*dudx + df4*dudy;
		#ifdef part_Inflammation
		if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
		{
			du[id] = ((D2*0.65 + (DD*this_x*this_x))*dudx2) + ((D2*0.65 + (DD*this_y*this_y))*dudy2) + 
			2 * (DD*this_x*this_y*dudxdy) +
			df1*dudx + df2*dudy + 
			df3*dudx + df4*dudy;
		}
		#endif
	}
}

__global__ void init_Istim(int num, double *u)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < num)
	{
		u[Istim + id*N1] = 0;
	}
}

__global__ void set_S1Istim(int totalNum, double *u, double strength, int *is_s1)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id < totalNum)
	{
		if(is_s1[id]) // position
		{
			u[id*N1 + Istim] = strength;
		}
	}
}

__global__ void set_S2Istim(int totalNum, double *u, double strength, int *is_s2,int *dev_rev_g,short *typeArray)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id < totalNum)
	{
		int host_id = dev_rev_g[id+1];
		int present_x = host_id/DimX;
		int present_y = host_id%DimX;
		short type = typeArray[id];
		if(present_x<=340 && present_x>=80 && present_y>=0 && present_y<=200)
		{
			if(type==3)
				u[id*N1 + Istim] = strength;
		}
	}
}


__global__ void Itot1(int totalNum, double dt, double *u, short *typeArray, double *Itotr, double *u_v,int *dev_rev_g)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	//printf("*******************\n");
	if (id < totalNum)
	{
		double k1,k2,kCaSR,dNai,dKi,dCai,dCaSR,dCaSS,dRR,Ek,Ena,Eks,Eca,CaCSQN,bjsr,cjsr,CaSSBuf,bcss,ccss,CaBuf,bc,cc,Ak1,Bk1;
		double rec_iK1,rec_ipK,rec_iNaK,AM,BM,AH_1,BH_1,AH_2,BH_2,AJ_1,BJ_1,AJ_2,BJ_2,M_INF,H_INF,J_INF,TAU_M,TAU_H,TAU_J,axr1,bxr1,axr2,bxr2,Xr1_INF,Xr2_INF;
		double TAU_Xr1,TAU_Xr2,Axs,Bxs,Xs_INF,TAU_Xs,R_INF,TAU_R,S_INF,TAU_S,Ad,Bd,Cd,Af,Bf,Cf,Af2,Bf2,Cf2,TAU_D,D_INF,TAU_F,F_INF,TAU_F2,F2_INF,TAU_FCaSS,FCaSS_INF;
		double IKr,IKs,IK1,Ito,INa,INaL,IbNa,ICaL,IbCa,INaCa,IpCa,IpK,INaK,Irel,Ileak,Iup,Ixfer,dvdt;
		double inverseVcF2 = 1/(2*Vc*F);
		double inverseVcF = 1./(Vc*F);
		double inversevssF2 = 1/(2*Vss*F);
		double RTONF = (R*T)/F;  
		short type = typeArray[id]; 
		double this_u_v = u_v[id];
		double Gkr = 0.153;
		double Gks;
		double Gto;

		int host_id = dev_rev_g[id+1];
		int present_x = host_id/DimX;
		int present_y = host_id%DimX;
		if(type==3) //EPI
		{
			Gks = 0.392;
			Gto = 0.294;
		}else if(type==2)//MCELL
		{
			Gks = 0.098;
			Gto = 0.294;
		}else{       //ENDO
			
			Gks = 0.392;
			Gto = 0.073;
		}
		
		
		#ifdef part_Inflammation
		if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
		{
			Gkr *= 0.37;
			Gto *= 0.4;
		}
		#endif
		

		
		#ifdef all_Inflammation
		Gkr *= 0.37;
		Gto *= 0.4;
		#endif
		


		Ek = RTONF*(log((Ko/u[id*N1+Ki])));
		Ena = RTONF*(log((Nao/u[id*N1+Nai])));
		Eks = RTONF*(log((Ko+pKNa*Nao)/(u[id*N1+Ki]+pKNa*u[id*N1+Nai])));
		Eca = 0.5*RTONF*(log((Cao/u[id*N1+Cai])));
		Ak1 = 0.1/(1.+exp(0.06*(this_u_v-Ek-200)));
		Bk1 = (3.*exp(0.0002*(this_u_v-Ek+100))+ exp(0.1*(this_u_v-Ek-10)))/(1.+exp(-0.5*(this_u_v-Ek)));
		rec_iK1 = Ak1/(Ak1+Bk1);
		rec_iNaK = (1./(1.+0.1245*exp(-0.1*this_u_v*F/(R*T))+0.0353*exp(-this_u_v*F/(R*T))));
		rec_ipK = 1./(1.+exp((25-this_u_v)/5.98));


		//Compute currents
		INa=GNa*u[id*N1+sm]*u[id*N1+sm]*u[id*N1+sm]*u[id*N1+sh]*u[id*N1+sj]*(this_u_v-Ena);
		ICaL = GCaL*u[id*N1+sd]*u[id*N1+sf]*u[id*N1+sf2]*u[id*N1+sfcass]*4*(this_u_v-15)*(F*F/(R*T))*
		  (0.25*exp(2*(this_u_v-15)*F/(R*T))*u[id*N1+CaSS]-Cao)/(exp(2*(this_u_v-15)*F/(R*T))-1.);
		  

		#ifdef part_Inflammation
		if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
		{
			ICaL *= 1.27;
		}
		#endif
		#ifdef all_Inflammation
			ICaL =ICaL *1.27;
		#endif
		
		

		Ito = Gto*u[id*N1+sr]*u[id*N1+ss]*(this_u_v-Ek);
		IKr = Gkr*sqrt(Ko/5.4)*u[id*N1+sxr1]*u[id*N1+sxr2]*(this_u_v-Ek);
		IKs = Gks*u[id*N1+sxs]*u[id*N1+sxs]*(this_u_v-Eks);
		IK1 = GK1*rec_iK1*(this_u_v-Ek); // different from Lu
		INaCa = knaca*(1./(KmNai*KmNai*KmNai+Nao*Nao*Nao))*(1./(KmCa+Cao))*
		  (1./(1+ksat*exp((n-1)*this_u_v*F/(R*T))))*
		  (exp(n*this_u_v*F/(R*T))*u[id*N1+Nai]*u[id*N1+Nai]*u[id*N1+Nai]*Cao-
		   exp((n-1)*this_u_v*F/(R*T))*Nao*Nao*Nao*u[id*N1+Cai]*2.5);
		INaK = knak*(Ko/(Ko+KmK))*(u[id*N1+Nai]/(u[id*N1+Nai]+KmNa))*rec_iNaK;
		IpCa = GpCa*u[id*N1+Cai]/(KpCa+u[id*N1+Cai]);
		IpK = GpK*rec_ipK*(this_u_v-Ek);
		IbNa = GbNa*(this_u_v-Ena);
		IbCa = GbCa*(this_u_v-Eca);


		//Determine total current   pA/pF
		Itotr[id] = -(IKr +
		  IKs   +
		  IK1   +
		  Ito   +
		  INa   +
		  IbNa  +
		  ICaL  +
		  IbCa  +
		  INaK  +
		  INaCa +
		  IpCa  +
		  IpK   +
		  u[id*N1+Istim]);

	 
		//update concentrations    
		kCaSR=maxsr-((maxsr-minsr)/(1+(EC/u[id*N1+CaSR])*(EC/u[id*N1+CaSR]))); 
		k1=k1_/kCaSR;
		k2=k2_*kCaSR;
		dRR=k4*(1-u[id*N1+sRR])-k2*u[id*N1+CaSS]*u[id*N1+sRR];
		u[id*N1+sRR]+=dt*dRR;
		u[id*N1+sOO]=k1*u[id*N1+CaSS]*u[id*N1+CaSS]*u[id*N1+sRR]/(k3+k1*u[id*N1+CaSS]*u[id*N1+CaSS]);


		Irel=Vrel*u[id*N1+sOO]*(u[id*N1+CaSR]-u[id*N1+CaSS]);
		Ileak=Vleak*(u[id*N1+CaSR]-u[id*N1+Cai]);

		#ifdef part_Inflammation
		if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
		{
			Ileak*=1.63;
		}
		#endif

		
		#ifdef all_Inflammation
		Ileak = Ileak *1.63;
		#endif
		
		Iup=Vmaxup/(1.+((Kup*Kup)/(u[id*N1+Cai]*u[id*N1+Cai])));
		
		
		#ifdef part_Inflammation
		if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
		{
			Iup = Iup *0.6;
		}
		#endif
		#ifdef all_Inflammation
		Iup = Iup *0.6;
		#endif
		Ixfer=Vxfer*(u[id*N1+CaSS]-u[id*N1+Cai]);

		CaCSQN=Bufsr*u[id*N1+CaSR]/(u[id*N1+CaSR]+Kbufsr);
		dCaSR=dt*(Iup-Irel-Ileak);
		bjsr=Bufsr-CaCSQN-dCaSR-u[id*N1+CaSR]+Kbufsr;
		cjsr=Kbufsr*(CaCSQN+dCaSR+u[id*N1+CaSR]);
		u[id*N1+CaSR]=(sqrt(bjsr*bjsr+4*cjsr)-bjsr)/2;
	   

		CaSSBuf=Bufss*u[id*N1+CaSS]/(u[id*N1+CaSS]+Kbufss);
		dCaSS=dt*(-Ixfer*(Vc/Vss)+Irel*(Vsr/Vss)+(-ICaL*inversevssF2*CAPACITANCE));
		bcss=Bufss-CaSSBuf-dCaSS-u[id*N1+CaSS]+Kbufss;
		ccss=Kbufss*(CaSSBuf+dCaSS+u[id*N1+CaSS]);
		u[id*N1+CaSS]=(sqrt(bcss*bcss+4*ccss)-bcss)/2;


		CaBuf=Bufc*u[id*N1+Cai]/(u[id*N1+Cai]+Kbufc);
		dCai=dt*((-(IbCa+IpCa-2*INaCa)*inverseVcF2*CAPACITANCE)-(Iup-Ileak)*(Vsr/Vc)+Ixfer);
		bc=Bufc-CaBuf-dCai-u[id*N1+Cai]+Kbufc;
		cc=Kbufc*(CaBuf+dCai+u[id*N1+Cai]);
		u[id*N1+Cai]=(sqrt(bc*bc+4*cc)-bc)/2;
			
		
		dNai = -(INa+IbNa+3*INaK+3*INaCa)*inverseVcF*CAPACITANCE;
		u[id*N1+Nai] += dt*dNai;
		
		dKi = -(u[id*N1+Istim]+IK1+Ito+IKr+IKs-2*INaK+IpK)*inverseVcF*CAPACITANCE;
		u[id*N1+Ki] += dt*dKi;



		//compute steady state values and time constants 
		AM=1./(1.+exp((-60.-this_u_v)/5.));
		BM=0.1/(1.+exp((this_u_v+35.)/5.))+0.10/(1.+exp((this_u_v-50.)/200.));
		TAU_M=AM*BM;
		M_INF=1./((1.+exp((-56.86-this_u_v)/9.03))*(1.+exp((-56.86-this_u_v)/9.03)));
		if (this_u_v>=-40.)
		{
			AH_1=0.; 
			BH_1=(0.77/(0.13*(1.+exp(-(this_u_v+10.66)/11.1))));
			TAU_H= 1.0/(AH_1+BH_1);
		}
		else
		{
			AH_2=(0.057*exp(-(this_u_v+80.)/6.8));
			BH_2=(2.7*exp(0.079*this_u_v)+(3.1e5)*exp(0.3485*this_u_v));
			TAU_H=1.0/(AH_2+BH_2);
		}
		H_INF=1./((1.+exp((this_u_v+71.55)/7.43))*(1.+exp((this_u_v+71.55)/7.43)));
		if(this_u_v>=-40.)
		{
			AJ_1=0.;      
			BJ_1=(0.6*exp((0.057)*this_u_v)/(1.+exp(-0.1*(this_u_v+32.))));
			TAU_J= 1.0/(AJ_1+BJ_1);
		}
		else
		{
			AJ_2=(((-2.5428e4)*exp(0.2444*this_u_v)-(6.948e-6)*
				exp(-0.04391*this_u_v))*(this_u_v+37.78)/
				  (1.+exp(0.311*(this_u_v+79.23))));    
			BJ_2=(0.02424*exp(-0.01052*this_u_v)/(1.+exp(-0.1378*(this_u_v+40.14))));
			TAU_J= 1.0/(AJ_2+BJ_2);
		}
		J_INF=H_INF;

		Xr1_INF=1./(1.+exp((-26.-this_u_v)/7.)); 

		#ifdef part_Inflammation
		if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
		{
			Xr1_INF=1./(1.+exp((-26.-(this_u_v-5))/7.));
		}
		#endif
		#ifdef all_Inflammation
		Xr1_INF=1./(1.+exp((-26.-(this_u_v-5))/7.));
		#endif

		axr1=450./(1.+exp((-45.-this_u_v)/10.));
		bxr1=6./(1.+exp((this_u_v-(-30.))/11.5));
		TAU_Xr1=axr1*bxr1;
		Xr2_INF=1./(1.+exp((this_u_v-(-88.))/24.));
		axr2=3./(1.+exp((-60.-this_u_v)/20.));
		bxr2=1.12/(1.+exp((this_u_v-60.)/20.));
		TAU_Xr2=axr2*bxr2;

		Xs_INF=1./(1.+exp((-5.-this_u_v)/14.));
		Axs=(1400./(sqrt(1.+exp((5.-this_u_v)/6))));
		Bxs=(1./(1.+exp((this_u_v-35.)/15.)));
		TAU_Xs=Axs*Bxs+80;
		
		if(type == 3) 
		{
			R_INF=1./(1.+exp((20-this_u_v)/6.)); 
			S_INF=1./(1.+exp((this_u_v+20)/5.)); 

			#ifdef part_Inflammation
			if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
			{
				S_INF=1./(1.+exp((this_u_v-5.7+20)/5.));
			}
			#endif
			#ifdef all_Inflammation
			S_INF=1./(1.+exp((this_u_v-5.7+20)/5.));
			#endif

			TAU_R=9.5*exp(-(this_u_v+40.)*(this_u_v+40.)/1800.)+0.8;
			TAU_S=85.*exp(-(this_u_v+45.)*(this_u_v+45.)/320.)+5./(1.+exp((this_u_v-20.)/5.))+3.;
		}
		else if(type == 1) 
		{
			R_INF=1./(1.+exp((20-this_u_v)/6.));
			S_INF=1./(1.+exp((this_u_v+28)/5.));
			#ifdef part_Inflammation
			if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
			{
				S_INF=1./(1.+exp((this_u_v-5.7+28)/5.));
			}
			#endif
			#ifdef all_Inflammation
			S_INF=1./(1.+exp((this_u_v-5.7+28)/5.));
			#endif
			

			TAU_R=9.5*exp(-(this_u_v+40.)*(this_u_v+40.)/1800.)+0.8;
			TAU_S=1000.*exp(-(this_u_v+67)*(this_u_v+67)/1000.)+8.;
		}
		else // MCELL
		{
			R_INF=1./(1.+exp((20-this_u_v)/6.));
			S_INF=1./(1.+exp((this_u_v+20)/5.));

			#ifdef part_Inflammation
			if(present_x<=370&&present_x>=240&&present_y>=0&&present_y<=200)
			{
				S_INF=1./(1.+exp((this_u_v-5.7+20)/5.));
			}
			#endif
			#ifdef all_Inflammation
			S_INF=1./(1.+exp((this_u_v-5.7+20)/5.));
			#endif
			TAU_R=9.5*exp(-(this_u_v+40.)*(this_u_v+40.)/1800.)+0.8;
			TAU_S=85.*exp(-(this_u_v+45.)*(this_u_v+45.)/320.)+5./(1.+exp((this_u_v-20.)/5.))+3.;
		}



		D_INF=1./(1.+exp((-8-this_u_v)/7.5)); // original
		Ad=1.4/(1.+exp((-35-this_u_v)/13))+0.25;
		Bd=1.4/(1.+exp((this_u_v+5)/5));
		Cd=1./(1.+exp((50-this_u_v)/20));
		TAU_D=Ad*Bd+Cd;

		F_INF=1./(1.+exp((this_u_v+20)/7)); // original

		Af=1102.5*exp(-(this_u_v+27)*(this_u_v+27)/225);
		Bf=200./(1+exp((13-this_u_v)/10.));
		Cf=(180./(1+exp((this_u_v+30)/10)))+20;
		TAU_F=Af+Bf+Cf;
		F2_INF=0.67/(1.+exp((this_u_v+35)/7))+0.33;
		Af2=600*exp(-(this_u_v+25)*(this_u_v+25)/170);
		Bf2=31/(1.+exp((25-this_u_v)/10));
		Cf2=16/(1.+exp((this_u_v+30)/10));
		TAU_F2=Af2+Bf2+Cf2;          //..........................
		FCaSS_INF=0.6/(1+(u[id*N1+CaSS]/0.05)*(u[id*N1+CaSS]/0.05))+0.4;
		TAU_FCaSS=80./(1+(u[id*N1+CaSS]/0.05)*(u[id*N1+CaSS]/0.05))+2.;
   


		//Update gates
		u[id*N1+sm] = M_INF-(M_INF-u[id*N1+sm])*exp(-dt/TAU_M);
		u[id*N1+sh] = H_INF-(H_INF-u[id*N1+sh])*exp(-dt/TAU_H);
		u[id*N1+sj] = J_INF-(J_INF-u[id*N1+sj])*exp(-dt/TAU_J);
		u[id*N1+sxr1] = Xr1_INF-(Xr1_INF-u[id*N1+sxr1])*exp(-dt/TAU_Xr1);
		u[id*N1+sxr2] = Xr2_INF-(Xr2_INF-u[id*N1+sxr2])*exp(-dt/TAU_Xr2);
		u[id*N1+sxs] = Xs_INF-(Xs_INF-u[id*N1+sxs])*exp(-dt/TAU_Xs);
		u[id*N1+ss] = S_INF-(S_INF-u[id*N1+ss])*exp(-dt/TAU_S);
		u[id*N1+sr] = R_INF-(R_INF-u[id*N1+sr])*exp(-dt/TAU_R);
		u[id*N1+sd] = D_INF-(D_INF-u[id*N1+sd])*exp(-dt/TAU_D); 
		u[id*N1+sf] = F_INF-(F_INF-u[id*N1+sf])*exp(-dt/TAU_F); 
		u[id*N1+sf2] = F2_INF-(F2_INF-u[id*N1+sf2])*exp(-dt/TAU_F2); 
		u[id*N1+sfcass] = FCaSS_INF-(FCaSS_INF-u[id*N1+sfcass])*exp(-dt/TAU_FCaSS);

	}

}


__global__ void new_u_v(int num, double *u_v, double dt, double *du, double *itotr)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id < num)
	{
		u_v[id] += dt*(du[id] + itotr[id]);
	}
}


int main(int argc, char **argv)
{
	HANDLE_ERROR(cudaSetDevice(1));
	int isbound;
	int gg[9];

	double root2 = sqrt(2.0);
	double ir, il, imax;
	double tflt;

	g = new int[ElementNum];    
	int *dev_g;


	HANDLE_ERROR(cudaMalloc((void**)&dev_g, sizeof(int)*ElementNum));

	for (int i = 0; i<ElementNum; i++)
	{
		g[i] = 0;
	}

	cout << "Reading file..." << endl;
	FILE *fp_ventricle;

	fp_ventricle = fopen("double_human_geometry_coordinate_with_e1.txt", "r");
	int num = 0;

	while (!feof(fp_ventricle))
	{
		int t1, t2, t3;
		double t4=0, t5=0;                                                  
		fscanf(fp_ventricle, "%d %d %lf %lf %d", &t1, &t2, &t5, &t4, &t3);
		g[(t1)*DimX + t2] = t3;
		if(t1 == 0 || t1 == X-1 || t2 == 0 || t2 == Y-1 )
			if (t3 != 0)  
				cout << "Tissue exists in geometry boundary!  " <<t1<<" , "<<t2<< endl;
		num = num + 1;
	}
	cout << "There are " << num << " ventricular points." << endl;
	cout.flush();
	fclose(fp_ventricle);

	int *rev_g = new int[num+1];
	int *dev_rev_g;
	HANDLE_ERROR(cudaMalloc((void**)&dev_rev_g, sizeof(int)*(num+1)));

	double *u, *dev_u;
	u = new double[num*N1];     
	
	/**
	FILE *readstatefile = fopen("OneDstates.dat","r");
	//定义二维数组，存储读取的状态变量值
	double **temparray =new double*[num];//
	for(int i =0;i<num;i++)
	{
		temparray[i] = new double[N1-1];
	}
	for(int i=0;i<num;i++)
	{
		for(int j=0;j<N1-1;j++)
		{
			fscanf(readstatefile,"%lf",&temparray[i][j]);
		}
	}
	fclose(readstatefile);
	
	for (int i = 0; i<num; i++)
	{
		//int temp_loc= (int)(i/Y);
		u[i*N1+ Cai   ] = temparray[i][0];
		u[i*N1+ CaSR  ] = temparray[i][1];
		u[i*N1+ CaSS  ] = temparray[i][2];
		u[i*N1+ Nai   ] = temparray[i][3];
		u[i*N1+ Ki    ] = temparray[i][4];
		u[i*N1+ sm    ] = temparray[i][5];
		u[i*N1+ sh    ] = temparray[i][6];
		u[i*N1+ sj    ] = temparray[i][7];
		u[i*N1+ sxr1  ] = temparray[i][8];
		u[i*N1+ sxr2  ] = temparray[i][9];
		u[i*N1+ sxs   ] = temparray[i][10];
		u[i*N1+ sr    ] = temparray[i][11];
		u[i*N1+ ss    ] = temparray[i][12];
		u[i*N1+ sd    ] = temparray[i][13];
		u[i*N1+ sf    ] = temparray[i][14];
		u[i*N1+ sf2   ] = temparray[i][15];
		u[i*N1+ sfcass] = temparray[i][16];                              
		u[i*N1+ sRR   ] = temparray[i][17];
		u[i*N1+ sOO   ] = temparray[i][18];
	}
	for(int i=0;i<num;i++)
	{
		delete []temparray[i];
		//temparray[i]=NULL;
	}
	delete []temparray;
	
	**/
	
	for (int i = 0; i<num; i++)
	{
		u[i*N1+ Cai   ] = 0.00007;
		u[i*N1+ CaSR  ] = 1.3;
		u[i*N1+ CaSS  ] = 0.00007;
		u[i*N1+ Nai   ] = 7.67;
		u[i*N1+ Ki    ] = 138.3;
		u[i*N1+ sm    ] = 0.;
		u[i*N1+ sh    ] = 0.75;
		u[i*N1+ sj    ] = 0.75;
		u[i*N1+ sxr1  ] = 0.;
		u[i*N1+ sxr2  ] = 1.;
		u[i*N1+ sxs   ] = 0.;
		u[i*N1+ sr    ] = 0.;
		u[i*N1+ ss    ] = 1.;
		u[i*N1+ sd    ] = 0.;
		u[i*N1+ sf    ] = 1.;
		u[i*N1+ sf2   ] = 1.;
		u[i*N1+ sfcass] = 1.;                              
		u[i*N1+ sRR   ] = 1.;
		u[i*N1+ sOO   ] = 0.;
	}
	cudaMalloc((void**)&dev_u, sizeof(double)*N1*num);
	int num1 = 1;

	for (int y = 0; y < Y; y++)
	{
		for (int x = 0; x < X; x++)
		{
			if (g[x*DimX + y]>0)
			{
				g[x*DimX + y] = num1; // start from 1 NOT 0 to make g>0;
				rev_g[num1] = x*DimX + y;
				num1++;
			}
		}
	}
	cudaMemcpy(dev_u, u, sizeof(double)*N1*num, cudaMemcpyHostToDevice);
	
	if (num != num1 - 1)
		cout << "Duplicated Points Found!" << endl;


	V_data = new double[num];  //double *V_data;

	double *xx, *dev_xx;
	xx = new double[num];
	HANDLE_ERROR(cudaMalloc((void**)&dev_xx, sizeof(double)*num));

	double *yy, *dev_yy;
	yy = new double[num];
	HANDLE_ERROR(cudaMalloc((void**)&dev_yy, sizeof(double)*num));


	short *type, *dev_type;
	type = new short[num];
	HANDLE_ERROR(cudaMalloc((void**)&dev_type, sizeof(short)*num));

	int *is_s1, *dev_is_s1;
	is_s1 = new int[num];
	HANDLE_ERROR(cudaMalloc((void**)&dev_is_s1, sizeof(int)*num));
	
	int *is_s2, *dev_is_s2;
	is_s2 = new int[num];
	 HANDLE_ERROR(cudaMalloc((void**)&dev_is_s2, sizeof(int)*num));

	double *dev_u_v;
	HANDLE_ERROR(cudaMalloc((void**)&dev_u_v, sizeof(double)*num));

	double *dev_du;
	HANDLE_ERROR(cudaMalloc((void**)&dev_du, sizeof(double)*num));

	double *dev_Itotr;
	HANDLE_ERROR(cudaMalloc((void**)&dev_Itotr, sizeof(double)*num));

	// test code. 
	double *Itotr = new double[num];
	double *du = new double[num]; 


	double *d;
	d = new double[num*5];
	double *dev_d;
	HANDLE_ERROR(cudaMalloc((void**)&dev_d, sizeof(double) * 5 * num));



	init_u_v << <(num + blockSize - 1) / blockSize, blockSize >> >(num, dev_u_v); 

	//read the stimulation file.
	FILE *sti_point;
	sti_point=fopen("double_geometry_stimulation_S1.dat","r");
	int coordinate[998][2];

	for(int i=0;i<998;i++)
	{
		fscanf(sti_point,"%d %d",&coordinate[i][0],&coordinate[i][1]);
	}
	fclose(sti_point);
	/**
	FILE *sti_S2;
	sti_S2=fopen("create_stimulation_point/160_180_stimulation_S2.dat","r");
	int coordinate_S2[3875][2];

	for(int i=0;i<3875;i++)
	{
		fscanf(sti_S2,"%d %d",&coordinate_S2[i][0],&coordinate_S2[i][1]);
		//printf("%d %d\n",coordinate_S2[i][0],coordinate_S2[i][1]);
	}
	fclose(sti_S2);
	**/
	fp_ventricle = fopen("double_human_geometry_coordinate_with_e1.txt", "r");
	cout << "Rescan ...";
	cout.flush();
	int first = 1; // test !!!!!
	while (!feof(fp_ventricle))
	{
		int t1, t2, t3;
		double t4=0, t5=0;
		fscanf(fp_ventricle, "%d %d %lf %lf %d", &t1, &t2, &t5, &t4, &t3);
		
		int index = g[t1*DimX + t2] - 1;

		int flag=1;
		for(int i=0;i<998;i++)
		{
			if(t1==coordinate[i][0]&&t2==coordinate[i][1])
			{
				is_s1[index] = 1;
				flag=0;
			}
		}
		if(flag)
			is_s1[index] = 0;
		/**
		int flag_S2=1;
		for(int i=0;i<3875;i++)
		{
			if(t1==coordinate_S2[i][0]&&t2==coordinate_S2[i][1])
			{
				//cout<<"set s2"<<endl;
				is_s2[index] = 1;
				flag_S2=0;
			}
		}
		if(flag_S2)
			is_s2[index] = 0;
		**/

		// test code. add double converter.
		xx[index] = t4;
		yy[index] = t5;
		type[index] = t3;
		
		//printf("%d\n",type[index]);
		// calculate D
		d[5*index+1] = (D1-D2)*t4*t4 + D2;
		d[5*index+2] = (D1-D2)*t4*t5 ;
		d[5*index+3] = (D1-D2)*t4*t5 ;
		d[5*index+4] = (D1-D2)*t5*t5 + D2;
	}
	
	#ifdef part_Inflammation
	for(int i=0;i<num;i++)
	{
		int temp_x = rev_g[i+1]/DimX;
		int temp_y = rev_g[i+1]%DimX;
		if(temp_x<=370&&temp_x>=240&&temp_y>=0&&temp_y<=200){
			d[5*i+1] = (D1-D2)*xx[i]*xx[i] + D2*0.65;
			d[5*i+2] = (D1-D2)*xx[i]*yy[i];
			d[5*i+3] = (D1-D2)*xx[i]*yy[i];
			d[5*i+4] = (D1-D2)*yy[i]*yy[i] + D2*0.65;
		}
	}
	#endif
	cout << "Done" << endl;
	cout.flush();
	fclose(fp_ventricle);

	HANDLE_ERROR(cudaMemcpy(dev_xx, xx, sizeof(double)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_yy, yy, sizeof(double)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_type, type, sizeof(short)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_is_s2, is_s2, sizeof(int)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_is_s1, is_s1, sizeof(int)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_d, d, sizeof(double) * 5 * num, cudaMemcpyHostToDevice));


	int i, j, k;

	// test coce. following 3 lines for approach 1.
	double dislow, dis;
	double normalx, normaly;
	int nnodex, nnodey, xcnt, ycnt;


			
	for (int y = 0; y<Y; y++)
	{			
		for (int x = 0; x<X; x++)
		{				
			if (g[x*DimX+y] == 0)
			{
				/*
				| 1 2 3 |
				| 4 s 5 |
				| 6 7 8 |
				*/
				if(x-1 >= 0 && y-1 >= 0 ) gg[1] = g[(x - 1)*DimX + (y - 1)];
				else gg[1] = 0;

				if(y-1 >= 0 ) gg[2] = g[x*DimX + y-1 ];
				else gg[2] = 0;

				if(x+1<X && y-1 >= 0 ) gg[3] = g[(x+1)*DimX + (y - 1)];
				else gg[3] = 0;

				if( x-1>=0) gg[4] = g[(x-1)*DimX +y];
				else gg[4] = 0;

				if(x+1<X) gg[5] = g[(x+1)*DimX + y];
				else gg[5] = 0;

				if(x-1 >=0 && y+1 < Y) gg[6] = g[(x-1)*DimX + (y+1)];
				else gg[6] = 0;

				if(y+1 < Y) gg[7] = g[x*DimX + y+1 ];
				else gg[7] = 0;

				if(x+1 < X && y+1 <Y) gg[8] = g[(x + 1)*DimX + (y + 1)];
				else gg[8] = 0;
									
			}


			isbound = 0;

			for (i = 1; i <= 8; i++)
			{
				if (gg[i]>0)
				{
					gg[i] = 1; isbound++;
				}
				else
				{
					gg[i] = 0;
				}
			}


			if (g[(x)*DimX + y] == 0 && isbound > 0)
			{
				/*
				| 1 2 3 |
				| 4 s 5 |
				| 6 7 8 |
				*/
				// approach 1 

				// il for x
				il = -gg[1]/root2 + gg[3]/root2 - gg[4] + gg[5] - gg[6]/root2 + gg[8]/root2;
				// ir for y
				ir = -gg[1]/root2 - gg[2] - gg[3]/root2 + gg[6]/root2 + gg[7] + gg[8]/root2;

				imax = fabs(ir);
				if (fabs(il) > imax)
					imax = fabs(il);
				

				// test code. 
				normalx = il/imax;
				normaly = ir/imax;

				if (imax-0 < 0.0001) {normalx = 0; normaly = 0; cout << "hit 000" << endl;} // imax is not int!!
				dislow = 1000;
				nnodex = 0; nnodey = 0; 
				for (ycnt=-1; ycnt<=1; ycnt++)
				for (xcnt=-1; xcnt<=1; xcnt++) 
				{
					dis = sqrt(1.0*(xcnt-normalx)*(xcnt-normalx)+(ycnt-normaly)*(ycnt-normaly));
					if (x+xcnt >= 0 && x+xcnt < X && y+ycnt >= 0 && y+ycnt < Y  )
					if (g[(x+xcnt)*DimX+(y+ycnt)] > 0 && dis < dislow) 
					{
						nnodex = xcnt;
						nnodey = ycnt;
						dislow = dis;
					}
				}
				g[x*DimX + y] = -g[(x+nnodex)*DimX+(y+nnodey)]; 
				// test code.
				if(g[x*DimX + y] == 0)  cout << "STILL GET NEIGHBOUR = 0";
			}
		}
	}


	HANDLE_ERROR(cudaMemcpy(dev_g, g, sizeof(int)*ElementNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_rev_g, rev_g, sizeof(int)*(num+1), cudaMemcpyHostToDevice));

	int *map = new int[num];
	int *dev_map;
	HANDLE_ERROR(cudaMalloc((void**)&dev_map, sizeof(int)*num));
	int cursor1 = 0;
	for (int i = 0; i<ElementNum; i++)
	{
		if (g[i] > 0)
		{
			map[cursor1] = i;
			++cursor1;
		}
	}
	HANDLE_ERROR(cudaMemcpy(dev_map, map, sizeof(int)*num, cudaMemcpyHostToDevice));

	//delete[] u;
	delete[] map;
	delete[] type;


	double time = 0;
	
	double BCL=1000;
	double numS1=1;
	double timedur =numS1*BCL;
	//double timedur =200;
	
	double stimstart=0;
	double stimduration=2;
	
	int count = 0;
	
	int countwrite=0;
	int flag[num];
	for(int i=0;i<num;i++)
	{
		flag[i]=1;
	}
	
	//statrt the circulation
	while (time <= timedur)
	{
		if (count%250 == 0)   
		{
			countwrite++;
			cout << "Progress = " << 100.0*time/timedur << "%" << endl;
			// WaitForSingleObjectEx(hWriteData, INFINITE, false);
			cudaMemcpy(V_data, dev_u_v, sizeof(double)*num, cudaMemcpyDeviceToHost);
			curCount = count;
			//writeData_ecg(time);
			writeData();
			// hWriteData = (HANDLE)_beginthreadex(NULL, 0, writeData, NULL, 0, NULL);
		}


		calc_du <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dev_d, dev_u_v, dev_type,dev_xx, dev_yy, dev_du, dev_map, dev_g, dev_rev_g);
		init_Istim <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dev_u);
		cudaDeviceSynchronize();

		//set stimulation
		if(time-floor(time/BCL)*BCL >= stimstart && time-floor(time/BCL)*BCL <= stimstart+stimduration)  // 350 early 370 late
		//if(time >= 0 && time <=2)
		{
			set_S1Istim<<<(num + blockSize - 1) / blockSize, blockSize>>>(num, dev_u, -52, dev_is_s1);
			cudaDeviceSynchronize();
		}

		/**
		if(time >= 347 && time < 349)      
		{
			set_S2Istim<<<(num + blockSize - 1) / blockSize, blockSize>>>(num, dev_u, -104, dev_is_s2,dev_rev_g,dev_type);
			cudaDeviceSynchronize();
		}
		**/
		
	
		




		Itot1 <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dt, dev_u, dev_type, dev_Itotr, dev_u_v, dev_rev_g);
		cudaDeviceSynchronize();

		new_u_v <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dev_u_v, dt, dev_du, dev_Itotr);
		cudaDeviceSynchronize();

		int oddnum = 0;
		int bignum = 0;
		int big_index = 0, small_index = 0;
		double vmin = 10000;
		double vmax = -1000;
		bool down_flag = false, up_flag = false;
		cudaMemcpy(V_data, dev_u_v, sizeof(double)*num, cudaMemcpyDeviceToHost);
		
		
		if(floor(time/BCL)==numS1-2 && floor((time+dt)/BCL)==numS1-1) 
		{ 
			char filename[30];
			sprintf(filename,"%lf_statefile.dat",time); 
			FILE *statefile = fopen(filename,"w+");
			cudaMemcpy(u, dev_u, sizeof(double)*N1*num, cudaMemcpyDeviceToHost);
			for (int i = 0; i<num; i++)
			{
				fprintf(statefile,"%4.10f\t",u[i*N1+ Cai   ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ CaSR  ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ CaSS  ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ Nai   ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ Ki    ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sm    ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sh    ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sj    ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sxr1  ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sxr2  ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sxs   ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sr    ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ ss    ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sd    ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sf    ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sf2   ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sfcass]);                       
				fprintf(statefile,"%4.10f\t",u[i*N1+ sRR   ]);
				fprintf(statefile,"%4.10f\t",u[i*N1+ sOO   ]);
				fprintf(statefile,"\n");
			}
			fclose(statefile);
		}
		

		for (i = 0; i < num; i++)
		{
			if(V_data[i]>-60)
			{
				flag[i]=0;
			}
			if(V_data[i] < -90)
			{
				down_flag = true;
				oddnum += 1;
				if(V_data[i] < vmin) {vmin = V_data[i]; small_index = i;}	
			}

			if(V_data[i] > 200)
			{
				up_flag = true;
				bignum += 1;
				if(V_data[i] > vmax) {vmax = V_data[i]; big_index = i;}
			}
		}
		
		if(up_flag) 
		{
			// cout << "ODD POINT NUMBER = " << oddnum << "; Vmin = " << vmin << "; time = " << time <<"; index = "<< index << endl;
			cout << "UP POINT NUMBER = " << bignum << "; Vmax = " << vmax << "; time = " << time <<"; index = "<< big_index << endl;
		}
		
		if(down_flag) 
		{
			cout << "DW POINT NUMBER = " << oddnum << "; Vmin = " << vmin << "; time = " << time <<"; index = "<< small_index << endl;
			// cout << "ODD POINT NUMBER = " << bignum << "; Vmax = " << vmax << "; time = " << time <<"; index = "<< index << endl;
		}
		
		time = time + dt;
		count++;
	}
	printf("totalwritenum: %d\n",countwrite);
	/**
	for(int i=0;i<num;i++)
	{
		if(flag[i])
		{
			for(int x=0;x<X;x++)
				for(int y=0;y<Y;y++)
				{
					if(g[x*DimX+y]==i+1)
					{
						cout<<i<<" "<<x<<" "<<y<<endl;
					}
				}
		}
	}
	**/

	delete[] g;
	//delete[] du;
	delete[] rev_g;
	delete[]V_data;
	delete[]u;
	

	HANDLE_ERROR(cudaFree(dev_g));
	HANDLE_ERROR(cudaFree(dev_rev_g));

	HANDLE_ERROR(cudaFree(dev_xx));
	HANDLE_ERROR(cudaFree(dev_yy));

	HANDLE_ERROR(cudaFree(dev_u_v));
	HANDLE_ERROR(cudaFree(dev_du));
	HANDLE_ERROR(cudaFree(dev_Itotr));
	// HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_map));
	HANDLE_ERROR(cudaFree(dev_type));
	HANDLE_ERROR(cudaFree(dev_u));
	HANDLE_ERROR(cudaFree(dev_d));
	HANDLE_ERROR(cudaFree(dev_is_s2));
	return 0;
}
