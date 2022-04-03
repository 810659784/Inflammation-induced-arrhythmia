#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <stdio.h>
#include <unistd.h> 
#include <sys/stat.h> 
#include <sys/types.h> 

#include <iomanip>

#include <cuda_runtime.h>
//#define local_Inflammation
//#define global_Inflammation
using namespace std;

__constant__ double R = 8314.5;
__constant__ double T = 295;
__constant__ double F = 96487;
__constant__ double Cm = 0.0001;
__constant__ double Vmyo = 25.85e3;
__constant__ double VSR = 2.098e3;
__constant__ double VmyouL = 25.85e-6;
__constant__ double VSRuL = 2.098e-6;
__constant__ double INaKmax = 0.95e-4;
__constant__ double KmK = 1.5;
__constant__ double KmNa_NaCa_NaK = 10;
__constant__ double Ko = 5.4;
__constant__ double Nao = 140;
__constant__ double CaTRPNMax = 70e-3;
__constant__ double gK1 = 0.024;
__constant__ double gf = 0.00145;
__constant__ double fNa = 0.2;
__constant__ double gBNa = 0.00008015;
__constant__ double gBCa = 0.0000324;
__constant__ double gBK = 0.000138;
//__constant__ double ECa = 65;
__constant__ double Cao = 1.2;
__constant__ double gD = 0.065;
__constant__ double JR = 0.02;
__constant__ double JL = 9.13e-4;
__constant__ double N = 50000;
__constant__ double KmNa_NaCa = 87.5;
__constant__ double KmCa = 1.38;
__constant__ double eta = 0.35;
__constant__ double ksat = 0.1;
__constant__ double gNCX = 38.5e-3;
__constant__ double gSERCA = 0.45e-3;
__constant__ double KSERCA = 0.5e-3;
__constant__ double gpCa = 0.0035e-3;
__constant__ double KmpCa = 0.5e-3;
__constant__ double gCaB = 2.6875e-8;
__constant__ double gSRl = 1.8951e-5;
__constant__ double kCMDN = 2.382e-3;
__constant__ double BCMDN = 50e-3;
__constant__ double kon = 100;
__constant__ double kRefoff = 0.2;
__constant__ double gammatrpn = 2;
__constant__ double alpha0 = 8e-3;
__constant__ double alphar1 = 2e-3;
__constant__ double alphar2 = 1.75e-3;
__constant__ double nRel = 3;
__constant__ double Kz = 0.15;
__constant__ double nHill = 3;
__constant__ double Ca50ref = 1.05e-3;
__constant__ double zp = 0.85;
__constant__ double beta1 = -4;
__constant__ double beta0 = 4.9;
__constant__ double Tref = 56.2;
__constant__ double a = 0.35;
__constant__ double A1 = -29;
__constant__ double A2 = 138;
__constant__ double A3 = 129;
__constant__ double alpha1 = 0.03;
__constant__ double alpha2 = 0.13;
__constant__ double alpha3 = 0.625;
__constant__ double VL = -2;
__constant__ double delVL = 7;
__constant__ double phiL = 2.35;
__constant__ double tL = 1;
__constant__ double tauL = 650;
__constant__ double tauR = 2.43;
__constant__ double phiR = 0.05;
__constant__ double thetaR = 0.012;
__constant__ double KRyR = 41e-3;
__constant__ double KL = 0.22e-3;
__constant__ double aCT = 0.0625;
__constant__ double bCT = 14;
__constant__ double cCT = 0.01;
__constant__ double dCT = 100;
__constant__ double dExtensionRatiodt = 0.00000;
__constant__ double tausss = 2.10000;



const int X = 115;
const int Y = 105;

const int DimX = Y;
const int ElementNum = X*Y;


const double D1 = 0.08;
const double D2 = 0.08;

//const double D1 = 0.08 * 0.65;
//const double D2 = 0.08 * 0.65;
const double DD = D1 - D2;




const double dx = 0.1;
const double dt = 0.005;

const int N1 = 22;

const int Istim = 0;
const int Nai =1 ;
const int Cai =2 ;
const int CaSR =3 ;
const int Ki =4 ;
const int TRPN =5 ;
const int m =6 ;
const int h =7 ;
const int j =8 ;
const int r =9 ;
const int s =10 ;
const int sslow =11 ;
const int rss =12 ;
const int sss = 13;
const int y = 14;
const int z = 15;
const int Q1 =16 ;
const int Q2 =17 ;
const int Q3 =18 ;
const int z1 =19 ;
const int z2 =20 ;
const int z3 = 21;



int *g;
double *V_data;
int curCount = 0;

const int blockSize = 128;

static void HandleError(cudaError_t err, const char *file, int line) {    
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " << file << " at line " << line << endl;
		char ch;
		cin >> ch;
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__))  

// test code.
void writeData()
{
	int file_id = curCount / 200;         
	std::ostringstream os;     
	os << "ap_" << file_id << ".vtk";
	std::ofstream out(os.str().c_str(), std::ios_base::out);

	out << "# vtk DataFile Version 3.0" << std::endl;
	out << "vtk output" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET STRUCTURED_POINTS" << std::endl;
	out << "DIMENSIONS " << X  << " " << Y  << " " << 1  << std::endl;
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
				/**
				if(time>0.2&&V_data[this_g - 1]>200)
				{
					printf("bug point is:\n");
					printf("x:%d y:%d z:%d\n",x,y,z);
					//cout<<"bug point is:"<<endl;
					//cout<<"x:"<<x<<" y:"<<y<<" z:"<<z<<endl;
				}
				**/
			}
			else
				out << "-100 ";
		}
		out << std::endl;
	}
	
	out.close();


}

void writeData_ecg()
{
	std::ostringstream os;      
	os << "2d_data";
	std::ofstream out(os.str().c_str(), std::ios_base::out|std::ios_base::app);

	
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
		u_v[id]=-80;
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
	if (_g > 0) // within tissue
	{
		// test code
		// temp = (int)(100000*u_v[_g - 1]);
		// return temp/100000.0;

		return u_v[_g - 1];
	}		
	else if (_g < 0) // boundary
	{
		return u_v[-_g - 1];

	}
	else // non-tissue
	{
		// test code.
		// return 1000000000;
		return -90;  // approach 2 
	}


}

// test code

double get_u_v_host(int host_id, int candidate_id, double *u_v, int *g)
{
	int _g = g[host_id];
	int temp;
	if (_g > 0)
	{
		// test code
		// temp = (int)(100000*u_v[_g - 1]);
		// return temp/100000.0;

		return u_v[_g - 1];
	}

	else if (_g < 0)
	{
		return u_v[-_g - 1]; 
	}

	else
	{
		// test code.
		// return 1000000000;
		return -90;
	}
	
}


__global__ void calc_du(int totalNum, double *d, double *u_v, double *xx, double *yy, double *du, int *map, int *g, int *dev_rev_g)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < totalNum)
	{
		// Glory modified. test code.
		int host_id = dev_rev_g[id+1];//map[id];
		double this_x = xx[id];
		double this_y = yy[id];
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
		//printf(".................\n");
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
		if(present_x<=85&&present_x>=50&&present_y>=0&&present_y<=50)
		{
			du[id] = ((D2*0.65 + (DD*this_x*this_x))*dudx2) + ((D2*0.65 + (DD*this_y*this_y))*dudy2) + 
			2 * (DD*0.65*this_x*this_y*dudxdy) +
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
		if(present_x<=114&&present_x>=70&&present_y>=0&&present_y<=35)
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
		double It,Iss,IK1,IBK,INaK,IfK,INa,IBNa,INaCa,IfNa,IRyR,ISERCA,ISR,If,ILCC,IpCa,ICaB,IBCa,IB,INaL;
		double sss_inf,m_inf,taum,h_inf,tauh,mL_inf,taumL,hL_inf,tauhL,j_inf,tauj,taur,r_inf,taus,s_inf,tausslow,sslow_inf,taurss,rss_inf,
			tauy,y_inf,EK,ENa,ECa,FVRT,tempINaCa,Cab,ExtensionRatio,lambda,Ca50,CaTRPN50,alphaTm,betaTm,overlap,zmax,TBase,
			T0,Q,Tension,koff,JTRPN,FVRTCa,Coc,mupoc,mupcc,expVL,alphap,betapcc,betapoc,denom,yoc,ycc,r1,mumoc,mumcc,
			r2,epsilonpcc,r7,epsilonm,r8,z4,Cco,epsilonpco,yco,r5,r6,r3,r4,yoo,JRco,JRoo,JR1,JR3,tempIRyR,JLoo,JLoc,
			JL1,JL2,tempILCC,tempIpCa,tempICaB,betaCMDN,Ccc,Coo,lambdaprev,yci,yoi,yic,yio,yii,dQ1dt,dQ2dt,dQ3dt,dsssdt;
		double dmdt,dhdt,dmLdt,dhLdt,djdt,drdt,dsdt,dsslowdt,drssdt,dydt,dKidt,dNaidt,dzdt,dTRPNdt,dz2dt,dz1dt,dz3dt,dCaSRdt,dCaidt;

		double gNa,gt,gss,aT0,bT0; 
		double sigma = (exp(Nao/67.3000) - 1.00000)/7.00000;
		double fK = 1.00000 - fNa;
		double K2 =  (( alphar2*pow(zp, nRel))/(pow(zp, nRel)+pow(Kz, nRel)))*(1.00000 - ( nRel*pow(Kz, nRel))/(pow(zp, nRel)+pow(Kz, nRel)));
		double tR =  1.17000*tL;
		double K1 = ( alphar2*pow(zp, nRel - 1.00000)*nRel*pow(Kz, nRel))/pow(pow(zp, nRel)+pow(Kz, nRel), 2.00000);
		double alpham = phiL/tL;
		double betam = phiR/tR;
		double this_u_v = u_v[id];
		int present_z=112;
		short type = typeArray[id];

		int host_id = dev_rev_g[id+1];
		int present_x = host_id/DimX;
		int present_y = host_id%DimX;

		if(type==2)//RV
		{
			gNa=0.8*1.33;
			gt=0.035*1.25;
			aT0=0.886;
			bT0=0.114;
			gss=0.007*1.1;
		}else if(type =3)//LV EPI
		{
			gNa = 0.8;
			gt = 0.035;
			gss= 0.007;
			aT0 = 0.886;
			bT0 = 0.114;
		}else{       //LV ENDO
			gNa = 1.33*0.8;
			gt = 0.4647*0.035;
			gss = 0.007;
			aT0 = 0.583;
			bT0= 0.417;
		}
		#ifdef local_Inflammation
		if(present_x<=85&&present_x>=50&&present_y>=0&&present_y<=50)
		{
			gt *= 0.4;
		}
		#endif

		#ifdef global_Inflammation
		gt*=0.4;
		#endif


		EK =  (( R*T)/F)*log(Ko/u[id*N1 + Ki]);
		It =  gt*u[id*N1 + r]*( aT0*u[id*N1 + s]+ bT0* u[id*N1 + sslow] )*(this_u_v - EK);
		r_inf = 1.00000/(1.00000+exp((this_u_v+10.6000)/- 11.4200));
		
		s_inf = 1.00000/(1.00000+exp((this_u_v+45.3000)/6.88410));
		sslow_inf = 1.00000/(1.00000+exp((this_u_v+45.3000)/6.88410));
		#ifdef local_Inflammation
		if(present_x<=85&&present_x>=50&&present_y>=0&&present_y<=50)
		{
			s_inf = 1.00000/(1.00000+exp(((this_u_v-5.7)+45.3000)/6.88410));
			sslow_inf = 1.00000/(1.00000+exp(((this_u_v-5.7)+45.3000)/6.88410));
		}
		#endif
		#ifdef global_Inflammation
		s_inf = 1.00000/(1.00000+exp(((this_u_v-5.7)+45.3000)/6.88410));
		sslow_inf = 1.00000/(1.00000+exp(((this_u_v-5.7)+45.3000)/6.88410));
		#endif
		
		taur = 1.00000/( 45.1600*exp( 0.0357700*(this_u_v+50.0000))+ 98.9000*exp( - 0.100000*(this_u_v+38.0000)));
		taus =  0.55*exp(- pow((this_u_v+70.0)/25.0, 2))+0.049;
		tausslow =  3.3*exp(( (- (this_u_v+70.0)/30.0)*(this_u_v+70.0))/30.0)+0.049;
		if(type == 3||type == 2)
		{
			taus = 0.35*exp(-pow((this_u_v+70.0)/15.0,2))+0.035;
			tausslow = 3.7*exp(-pow((this_u_v+70.0)/30.0,2))+0.035;
		}
		//...................Iss.................................
		Iss =  gss*u[id*N1 + rss  ]*u[id*N1 + sss  ]*(this_u_v - EK);
		rss_inf = 1.00000/(1.00000+exp((this_u_v+11.5000)/- 11.8200));
		sss_inf = 1.00000/(1.00000+exp((this_u_v+87.5000)/10.3000));
		taurss = 10.0000/( 45.1600*exp( 0.0357700*(this_u_v+50.0000))+ 98.9000*exp( - 0.100000*(this_u_v+38.0000)));

		//...................inward rectifier K+ current....................
		IK1 = ( (48.0000/(exp((this_u_v+37.0000)/25.0000)+exp((this_u_v+37.0000)/- 25.0000))+10.0000)*0.00100000)/(1.00000+exp((this_u_v - (EK+76.7700))/- 17.0000))+( gK1*(this_u_v - (EK+1.73000)))/( (1.00000+exp(( 1.61300*F*(this_u_v - (EK+1.73000)))/( R*T)))*(1.00000+exp((Ko - 0.998800)/- 0.124000)));
		IBK =  gBK*(this_u_v - EK);
		INaK = ( (( (( INaKmax*1.00000)/(1.00000+ 0.124500*exp(( - 0.100000*this_u_v*F)/( R*T))+ 0.0365000*sigma*exp(( - this_u_v*F)/( R*T))))*Ko)/(Ko+KmK))*1.00000)/(1.00000+pow(KmNa_NaCa_NaK/u[id*N1 + Nai], 4.00000));
		IfK =  gf*u[id*N1 + y    ]*fK*(this_u_v - EK);
		ENa =  (( R*T)/F)*log(Nao /u[id*N1 + Nai]);
		INa =  gNa*pow(u[id*N1 + m], 3.00000)*u[id*N1 + h]*u[id*N1 + j]*(this_u_v - ENa);

		m_inf = 1.00000/(1.00000+exp((this_u_v+45.0000)/- 6.50000));
		h_inf = 1.00000/(1.00000+exp((this_u_v+76.1000)/6.07000));
		j_inf = 1.00000/(1.00000+exp((this_u_v+76.1000)/6.07000));
		y_inf = 1.00000/(1.00000+exp((this_u_v+138.600)/10.4800));

		taum = 0.00136000/(( 0.320000*(this_u_v+47.1300))/(1.00000 - exp( - 0.100000*(this_u_v+47.1300)))+ 0.0800000*exp(- this_u_v/11.0000));
		tauh = (this_u_v>=- 40.0000 ?  0.000453700*(1.00000+exp(- (this_u_v+10.6600)/11.1000)) : 0.00349000/( 0.135000*exp(- (this_u_v+80.0000)/6.80000)+ 3.56000*exp( 0.0790000*this_u_v)+ 310000.*exp( 0.350000*this_u_v)));
		tauj = (this_u_v>=- 40.0000 ? ( 0.0116300*(1.00000+exp( - 0.100000*(this_u_v+32.0000))))/exp( - 2.53500e-07*this_u_v) : 0.00349000/( ((this_u_v+37.7800)/(1.00000+exp( 0.311000*(this_u_v+79.2300))))*( - 127140.*exp( 0.244400*this_u_v) -  3.47400e-05*exp( - 0.0439100*this_u_v))+( 0.121200*exp( - 0.0105200*this_u_v))/(1.00000+exp( - 0.137800*(this_u_v+40.1400)))));
		tauy = 1.00000/( 0.118850*exp((this_u_v+80.0000)/28.3700)+ 0.562300*exp((this_u_v+80.0000)/- 14.1900));

		IBNa =  gBNa*(this_u_v - ENa);
		FVRT = ( F*this_u_v)/( R*T);

		//.........Na-Ca exchanger,INaCa
		tempINaCa = ( gNCX*( exp( eta*FVRT)*pow(u[ id *N1 + Nai  ], 3.00000)*Cao -  exp( (eta - 1.00000)*FVRT)*pow(Nao, 3.00000)*u[Cai  +id*N1]))/( (pow(Nao, 3.00000)+pow(KmNa_NaCa, 3.00000))*(Cao+KmCa)*(1.00000+ ksat*exp( (eta - 1.00000)*FVRT)));
		INaCa =  tempINaCa*VmyouL*F;
		IfNa =  gf*u[id * N1 + y    ]*fNa*(this_u_v - ENa);
		Cab = CaTRPNMax - u[id*N1 + TRPN ];
		ExtensionRatio =1;//(VOI>300000. ? 1.00000 : 1.00000);
		lambda = (ExtensionRatio>0.800000&&ExtensionRatio<=1.15000 ? ExtensionRatio : ExtensionRatio>1.15000 ? 1.15000 : 0.800000);
		Ca50 =  Ca50ref*(1.00000+ beta1*(lambda - 1.00000));
		CaTRPN50 = ( Ca50*CaTRPNMax)/(Ca50+ (kRefoff/kon)*(1.00000 - ( (1.00000+ beta0*(lambda - 1.00000))*0.500000)/gammatrpn));
		alphaTm =  alpha0*pow(Cab/CaTRPN50, nHill);
		betaTm = alphar1+( alphar2*pow(u[id*N1 + z    ], nRel - 1.00000))/(pow(u[id*N1 + z    ], nRel)+pow(Kz, nRel));
		overlap = 1.00000+ beta0*(lambda - 1.00000);
		zmax = (alpha0/pow(CaTRPN50/CaTRPNMax, nHill) - K2)/(alphar1+K1+alpha0/pow(CaTRPN50/CaTRPNMax, nHill));
		TBase = ( Tref*u[id*N1 + z    ])/zmax;
		T0 =  TBase*overlap;
		Q = u[id*N1 + Q1   ]+u[id*N1 + Q2   ]+u[id*N1 + Q3   ];
		Tension = (Q<0.00000 ? ( T0*( a*Q+1.00000))/(1.00000 - Q) : ( T0*(1.00000+ (a+2.00000)*Q))/(1.00000+Q));
		koff = (1.00000 - Tension/( gammatrpn*Tref)>0.100000 ?  kRefoff*(1.00000 - Tension/( gammatrpn*Tref)) :  kRefoff*0.100000);
		JTRPN =  (CaTRPNMax - u[id*N1 + TRPN ])*koff -  u[Cai  +id*N1]*u[id*N1 + TRPN ]*kon;
		FVRTCa =  2.00000*FVRT;
		Coc = (fabs(FVRTCa)>1.00000e-09 ? (u[Cai  +id*N1]+( (JL/gD)*Cao*FVRTCa*exp(- FVRTCa))/(1.00000 - exp(- FVRTCa)))/(1.00000+( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa))) : (u[id*N1 + Cai  ]+ (JL/gD)*Cao)/(1.00000+JL/gD));
		mupoc = (pow(Coc, 2.00000)+ cCT*pow(KRyR, 2.00000))/( tauR*(pow(Coc, 2.00000)+pow(KRyR, 2.00000)));
		mupcc = (pow(u[Cai  +id*N1], 2.00000)+ cCT*pow(KRyR, 2.00000))/( tauR*(pow(u[Cai  +id*N1], 2.00000)+pow(KRyR, 2.00000)));
		expVL = exp((this_u_v - VL)/delVL);
		alphap = expVL/( tL*(expVL+1.00000));
		betapcc = pow(u[Cai  +id*N1], 2.00000)/( tR*(pow(u[Cai  +id*N1], 2.00000)+pow(KRyR, 2.00000)));
		betapoc = pow(Coc, 2.00000)/( tR*(pow(Coc, 2.00000)+pow(KRyR, 2.00000)));
		denom =  (alphap+alpham)*( (alpham+betam+betapoc)*(betam+betapcc)+ alphap*(betam+betapoc));
		yoc = ( alphap*betam*(alphap+alpham+betam+betapcc))/denom;
		ycc = ( alpham*betam*(alpham+alphap+betam+betapoc))/denom;
		r1 =  yoc*mupoc+ ycc*mupcc;
		mumoc = ( thetaR*dCT*(pow(Coc, 2.00000)+ cCT*pow(KRyR, 2.00000)))/( tauR*( dCT*pow(Coc, 2.00000)+ cCT*pow(KRyR, 2.00000)));
		mumcc = ( thetaR*dCT*(pow(u[Cai  +id*N1], 2.00000)+ cCT*pow(KRyR, 2.00000)))/( tauR*( dCT*pow(u[Cai  +id*N1], 2.00000)+ cCT*pow(KRyR, 2.00000)));
		r2 = ( alphap*mumoc+ alpham*mumcc)/(alphap+alpham);
		epsilonpcc = ( u[Cai  +id*N1]*(expVL+aCT))/( tauL*KL*(expVL+1.00000));
		r7 = ( alpham*epsilonpcc)/(alphap+alpham);
		epsilonm = ( bCT*(expVL+aCT))/( tauL*( bCT*expVL+aCT));
		r8 = epsilonm;
		z4 = ((1.00000 - u[id*N1 + z1   ]) - u[id*N1 + z2   ]) - u[id*N1 + z3   ];
		Cco = (u[Cai  +id*N1]+ (JR/gD)*u[id*N1 + CaSR ])/(1.00000+JR/gD);
		epsilonpco = ( Cco*(expVL+aCT))/( tauL*KL*(expVL+1.00000));
		yco = ( alpham*( betapcc*(alpham+betam+betapoc)+ betapoc*alphap))/denom;
		r5 =  yco*epsilonpco+ ycc*epsilonpcc;
		r6 = epsilonm;
		r3 = ( betam*mupcc)/(betam+betapcc);
		r4 = mumcc;
		yoo = ( alphap*( betapoc*(alphap+betam+betapcc)+ betapcc*alpham))/denom;
		JRco = ( JR*(u[id*N1 + CaSR ] - u[Cai  +id*N1]))/(1.00000+JR/gD);
		JRoo = (fabs(FVRTCa)>1.00000e-05 ? ( JR*((u[id*N1 + CaSR ] - u[Cai  +id*N1])+ (( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa)))*(u[id*N1 + CaSR ] -  Cao*exp(- FVRTCa))))/(1.00000+JR/gD+( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa))) : ( JR*((u[id*N1 + CaSR ] - u[id*N1 + Cai  ])+ (( (JL/gD)*1.00000e-05)/(1.00000 - exp(- 1.00000e-05)))*(u[id*N1 + CaSR ] -  Cao*exp(- 1.00000e-05))))/(1.00000+JR/gD+( (JL/gD)*1.00000e-05)/(1.00000 - exp(- 1.00000e-05))));
		JR1 =  yoo*JRoo+ JRco*yco;
		JR3 = ( JRco*betapcc)/(betam+betapcc);
		tempIRyR = ( ( u[id*N1 + z1   ]*JR1+ u[id*N1 + z3   ]*JR3)*N)/Vmyo;
		IRyR =  1.50000*tempIRyR;
		
		ISERCA = ( gSERCA*pow(u[Cai  +id*N1], 2.00000))/(pow(KSERCA, 2.00000)+pow(u[Cai  +id*N1], 2.00000));
		ISR =  gSRl*(u[id*N1 + CaSR ] - u[Cai  +id*N1]);
		#ifdef local_Inflammation
		if(present_x<=85&&present_x>=50&&present_y>=0&&present_y<=50){
			ISERCA *= 0.8;
			ISR *= 1.63;
		}
		#endif
		
		#ifdef global_Inflammation
		ISERCA *= 0.8;
		ISR *= 1.63;
		#endif
		
		If = IfNa+IfK;
		
		JLoo = (fabs(FVRTCa)>1.00000e-05 ? ( (( JL*FVRTCa)/(1.00000 - exp(- FVRTCa)))*(( Cao*exp(- FVRTCa) - u[Cai  +id*N1])+ (JR/gD)*( Cao*exp(- FVRTCa) - u[id*N1 + CaSR ])))/(1.00000+JR/gD+( (JL/gD)*FVRTCa)/(1.00000 - exp(FVRTCa))) : ( (( JL*1.00000e-05)/(1.00000 - exp(- 1.00000e-05)))*(( Cao*exp(- 1.00000e-05) - u[id*N1 + Cai  ])+ (JR/gD)*( Cao*exp(- 1.00000e-05) - u[id*N1 + CaSR ])))/(1.00000+JR/gD+( (JL/gD)*1.00000e-05)/(1.00000 - exp(- 1.00000e-05))));
		JLoc = (fabs(FVRTCa)>1.00000e-05 ? ( (( JL*FVRTCa)/(1.00000 - exp(- FVRTCa)))*( Cao*exp(- FVRTCa) - u[Cai  +id*N1]))/(1.00000+( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa))) : ( (( JL*1.00000e-05)/(1.00000 - exp(- 1.00000e-05)))*( Cao*exp(- 1.00000e-05) - u[id*N1 + Cai  ]))/(1.00000+( (JL/gD)*1.00000e-05)/(1.00000 - exp(- 1.00000e-05))));
		JL1 =  JLoo*yoo+ JLoc*yoc;
		JL2 = ( JLoc*alphap)/(alphap+alpham);
		tempILCC = ( ( u[id*N1 + z1   ]*JL1+ u[id*N1 + z2   ]*JL2)*N)/Vmyo;
		ILCC =  - 1.50000*tempILCC*2.00000*VmyouL*F;
		ILCC *= 0.48;
		if(type==2)
		{
			ILCC *=1.1;
		}

		#ifdef local_Inflammation
		if(present_x<=85&&present_x>=50&&present_y>=0&&present_y<=50){
			ILCC *=1.27;
		}
		#endif
		#ifdef global_Inflammation
		ILCC *=1.27;
		#endif

		tempIpCa = ( gpCa*u[Cai  +id*N1])/(KmpCa+u[Cai  +id*N1]);
		IpCa =  tempIpCa*2.00000*VmyouL*F;
		
		ECa =(( R*T)/( 2.00000*F))*log( Cao / u[id*N1+Cai]);
		
		tempICaB =  gCaB*(ECa - this_u_v);
		ICaB =  - tempICaB*2.00000*VmyouL*F;
		betaCMDN = pow(1.00000+( kCMDN*BCMDN)/pow(kCMDN+u[Cai  +id*N1], 2.00000), - 1.00000);

		Ccc = u[Cai  +id*N1];
		Coo = (fabs(FVRTCa)>1.00000e-09 ? (u[Cai  +id*N1]+ (JR/gD)*u[id*N1 + CaSR ]+( (JL/gD)*Cao*FVRTCa*exp(- FVRTCa))/(1.00000 - exp(- FVRTCa)))/(1.00000+JR/gD+( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa))) : (u[id*N1 + Cai  ]+ (JR/gD)*u[id*N1 + CaSR ]+ (JL/gD)*Cao)/(1.00000+JR/gD+JL/gD));
		IBCa =  gBCa*(this_u_v - ECa);
		IB = IBNa+IBCa+IBK;
		lambdaprev = ExtensionRatio;
		yci = alpham/(alphap+alpham);
		yoi = alphap/(alphap+alpham);
		yic = betam/(betapcc+betam);
		yio = betapcc/(betapcc+betam);
		yii = (((((((1.00000 - yoc) - yco) - yoo) - ycc) - yci) - yic) - yoi) - yio;
		
		dQ1dt =  A1*dExtensionRatiodt -  alpha1* u[id*N1 + Q1   ];
		dQ2dt =  A2*dExtensionRatiodt -  alpha2* u[id*N1 + Q2   ];
		dQ3dt =  A3*dExtensionRatiodt -  alpha3* u[id*N1 + Q3   ];
		dsssdt = ((sss_inf - u[id*N1 + sss  ])/tausss)*0.00100000;
		
		
		dmdt = ((m_inf - u[id*N1 + m ])/taum)*0.00100000;
		dhdt = ((h_inf - u[id*N1 + h ])/tauh)*0.00100000;
		djdt = ((j_inf - u[id*N1 + j ])/tauj)*0.00100000;
		drdt = ((r_inf - u[id*N1 + r ])/taur)*0.00100000;
		dsdt = ((s_inf - u[id*N1 + s ])/taus)*0.00100000;
		dsslowdt = ((sslow_inf - u[id*N1 + sslow])/tausslow)*0.001;
		drssdt = ((rss_inf - u[id*N1 + rss])/taurss)*0.00100000;
		dydt = ((y_inf - u[id*N1 + y])/tauy)*0.00100000;
		dKidt = ( - (u[id*N1 + Istim]+Iss*0.00100000+IBK*0.00100000+It*0.00100000+IK1*0.00100000+IfK*0.00100000+ INaK*- 2.00000)*1.00000)/( VmyouL*F);
		dNaidt = ( - (INa*0.00100000+IBNa*0.00100000+ INaCa*3.00000+ INaK*3.00000+IfNa*0.00100000)*1.00000)/( VmyouL*F);
		dzdt =  alphaTm*(1.00000 - u[id*N1 + z]) -  betaTm*u[id*N1 + z];
		dTRPNdt = JTRPN;
		dz2dt = ( r1 * u[id*N1 + z1] -  (r2+r7) * u[id*N1 + z2])+ r8*z4;
		dz1dt =  - (r1+r5) * u[id*N1 + z1] + r2 * u[id*N1 + z2] + r6 * u[id*N1 + z3];
		dz3dt = ( r5 * u[id*N1 + z1] -  (r6+r3)*u[id*N1 + z3])+ r4*z4;
		dCaSRdt =  (VmyouL/VSRuL)*((- IRyR+ISERCA) - ISR);
		Itotr[id] = - (INa*0.00100000+It*0.00100000+Iss*0.00100000+If*0.00100000+IK1*0.00100000+IBNa*0.00100000+IBK*0.00100000+INaK+u[id*N1 + Istim]+ICaB+INaCa+IpCa+ILCC)/Cm;

		dCaidt =  betaCMDN*(((IRyR - ISERCA)+ISR+JTRPN) - ( - 2.00000*INaCa+IpCa+ICaB+ILCC)/( 2.00000*VmyouL*F));

		u[id*N1 + m    ]   += dt*dmdt;// is d/dt m in component sodium_current_m_gate (dimensionless).
		u[id*N1 + h    ]   += dt*dhdt;// is d/dt h in component sodium_current_h_gate (dimensionless).
		u[id*N1 + j    ]   += dt*djdt;// is d/dt j in component sodium_current_j_gate (dimensionless).
		u[id*N1 + r    ]   += dt*drdt;// is d/dt r in component Ca_independent_transient_outward_K_current_r_gate (dimensionless).
		u[id*N1 + s    ]   += dt*dsdt;// is d/dt s in component Ca_independent_transient_outward_K_current_s_gate (dimensionless).
		u[id*N1+sslow]   += dt*dsslowdt;// is d/dt s_slow in component Ca_independent_transient_outward_K_current_s_slow_gate (dimensionless).
		u[id*N1+ rss]   += dt*drssdt;// is d/dt r_ss in component steady_state_outward_K_current_r_ss_gate (dimensionless).
		u[id*N1 + sss]   += dt*dsssdt;// is d/dt s_ss in component steady_state_outward_K_current_s_ss_gate (dimensionless).
		u[id*N1 + z1]   += dt*dz1dt;// is d/dt z_1 in component CaRU_reduced_states (dimensionless).
		u[id*N1 + y]   += dt*dydt;// is d/dt y in component hyperpolarisation_activated_current_y_gate (dimensionless).
		u[id*N1 + z2]   += dt*dz2dt;// is d/dt z_2 in component CaRU_reduced_states (dimensionless).
		u[id*N1 + z3]   += dt*dz3dt;// is d/dt z_3 in component CaRU_reduced_states (dimensionless).
		u[id*N1 + Nai  ]   += dt*dNaidt;// is d/dt Na_i in component intracellular_ion_concentrations (mM).
		u[id*N1 + Ki   ]   += dt*dKidt;// is d/dt K_i in component intracellular_ion_concentrations (mM).
		u[id*N1 + TRPN ]   += dt*dTRPNdt;// is d/dt TRPN in component intracellular_ion_concentrations (mM).
		u[id*N1 + Cai  ]   += dt*dCaidt;// is d/dt Ca_i in component intracellular_ion_concentrations (mM).
		u[id*N1 + CaSR ]   += dt*dCaSRdt;// is d/dt Ca_SR in component intracellular_ion_concentrations (mM).
		u[id*N1 + z    ]   += dt*dzdt;// is d/dt z in component tropomyosin (dimensionless).
		u[id*N1 + Q1   ]   += dt*dQ1dt;// is d/dt Q_1 in component Cross_Bridges (dimensionless).
		u[id*N1 + Q2   ]   += dt*dQ2dt;// is d/dt Q_2 in component Cross_Bridges (dimensionless).
		u[id*N1 + Q3   ]   += dt*dQ3dt;// is d/dt Q_3 in component Cross_Bridges (dimensionless).
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
	HANDLE_ERROR(cudaSetDevice(0));
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

	fp_ventricle = fopen("2D_160_180.txt", "r");
	int num = 0;

	while (!feof(fp_ventricle))
	{
		int t1, t2, t3, t7;
		
		float t4, t5;                                                   
		
		fscanf(fp_ventricle, "%d %d %d %f %f %d", &t1, &t2, &t3, &t4, &t5, &t7);
		if((t1==29&&t2==78)||(t1==30&&t2==78))
		{
		}else
		{
			g[(t1)*DimX + t2] = t7;
			if(t1 == 0 || t1 == X-1 || t2 == 0 || t2 == Y-1 )
				if (t7 != 0)  cout << "Tissue exists in geometry boundary!" << endl;
			num = num + 1;
		}
	}
	cout << "There are " << num << " ventricular points." << endl;
	cout.flush();
	fclose(fp_ventricle);

	// test code.
	int *rev_g = new int[num+1];
	int *dev_rev_g;
	HANDLE_ERROR(cudaMalloc((void**)&dev_rev_g, sizeof(int)*(num+1)));

	double *u, *dev_u;
	u = new double[num*N1];     //const int N = 26;
	for (int i = 0; i<num; i++)
	{
		u[Nai  +i*N1] =10.73519;
		u[Cai  +i*N1] =0.00007901351;
		u[CaSR +i*N1] =700.0e-3;
		u[Ki   +i*N1] =139.2751;
		u[TRPN +i*N1] =0.067593139865;
		u[m    +i*N1] =0.004164108;
		u[h    +i*N1] =0.6735613;
		u[j    +i*N1] =0.6729362;
		u[r    +i*N1] =0.002191519;
		u[s    +i*N1] =0.9842542;
		u[sslow+i*N1] =0.6421196;
		u[rss  +i*N1] =0.002907171;
		u[sss  +i*N1] =0.3142767;
		u[y    +i*N1] =0.003578708;
		u[z    +i*N1] =0.014417937837;
		u[Q1   +i*N1] =0;
		u[Q2   +i*N1] =0;
		u[Q3   +i*N1] =0;
		u[z1   +i*N1] =0.98859;
		u[z2   +i*N1] =0.0087302;
		u[z3   +i*N1] =0.0026566;
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
	sti_point=fopen("stimulation_point.dat","r");
	int coordinate[199][2];

	for(int i=0;i<199;i++)
	{
		fscanf(sti_point,"%d %d",&coordinate[i][0],&coordinate[i][1]);
	}
	fclose(sti_point);
	
	fp_ventricle = fopen("2D_160_180.txt", "r");
	cout << "Rescan ...";
	cout.flush();
	int first = 1; // test !!!!!
	while (!feof(fp_ventricle))
	{
		int t1, t2, t3, t7;
		double t4, t5;
		fscanf(fp_ventricle, "%d %d %d %lf %lf %d", &t1, &t2, &t3, &t4, &t5, &t7);
		if((t1==29&&t2==78)||(t1==30&&t2==78))
		{
		}else
		{
			int index = g[t1*DimX + t2] - 1;
			int flag=1;
			for(int i=0;i<199;i++)
			{
				if(t1==coordinate[i][0]&&t2==coordinate[i][1])
				{
					//cout<<"set s2"<<endl;
					is_s1[index] = 1;
					//printf("cccccccccc\n");
					flag=0;
				}
			}
			if(flag)
				is_s1[index] = 0;
			
			// test code. add double converter.
			xx[index] = t4;
			yy[index] = t5;
			type[index] = t7;
			// calculate D
			d[5*index+1] = (D1-D2)*t4*t4 + D2;
			d[5*index+2] = (D1-D2)*t4*t5 ;
			d[5*index+3] = (D1-D2)*t4*t5 ;
			d[5*index+4] = (D1-D2)*t5*t5 + D2;
		}
	}
	#ifdef part_Inflammation
	for(int i=0;i<num;i++)
	{
		int temp_x = rev_g[i+1]/DimX;
		int temp_y = rev_g[i+1]%DimX;
		if(present_x<=85&&present_x>=50&&present_y>=0&&present_y<=50){
			d[5*i+1] = (D1-D2)*0.65*xx[i]*xx[i] + D2*0.65;
			d[5*i+2] = (D1-D2)*0.65*xx[i]*yy[i];
			d[5*i+3] = (D1-D2)*0.65*xx[i]*yy[i];
			d[5*i+4] = (D1-D2)*0.65*yy[i]*yy[i] + D2*0.65;
		}
	}
	#endif
	cout << "Done" << endl;
	cout.flush();
	fclose(fp_ventricle);

	HANDLE_ERROR(cudaMemcpy(dev_xx, xx, sizeof(double)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_yy, yy, sizeof(double)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_type, type, sizeof(short)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_is_s2, is_s2, sizeof(int)*num, cudaMemcpyHostToDevice));//.........................._stimulation
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

	delete[] u;
	delete[] map;
	delete[] type;


	double time = 0;
	double BCL=1000;
	double numS1=2;
	double timedur =numS1*BCL;
	
	double stimstart=0;
	double stimduration=5;
	
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
		if (count%200 == 0)   
		{
			countwrite++;
			cout << "Progress = " << 100.0*time/timedur << "%" << endl;
			// WaitForSingleObjectEx(hWriteData, INFINITE, false);
			cudaMemcpy(V_data, dev_u_v, sizeof(double)*num, cudaMemcpyDeviceToHost);
			curCount = count;
			//writeData_ecg();
			writeData();
			// hWriteData = (HANDLE)_beginthreadex(NULL, 0, writeData, NULL, 0, NULL);
		}


		calc_du <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dev_d, dev_u_v, dev_xx, dev_yy, dev_du, dev_map, dev_g, dev_rev_g);
		init_Istim <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dev_u);
		cudaDeviceSynchronize();

		//set stimulation
		if(time-floor(time/BCL)*BCL >= stimstart && time-floor(time/BCL)*BCL <= stimstart+stimduration)  // 350 early 370 late
		//if(time >= 0 && time <= 5)
		{
			set_S1Istim<<<(num + blockSize - 1) / blockSize, blockSize>>>(num, dev_u, -0.008, dev_is_s1);
			cudaDeviceSynchronize();
		}
		/**
		if(time >= 64 && time <= 66)  
		{
			set_S2Istim<<<(num + blockSize - 1) / blockSize, blockSize>>>(num, dev_u, -0.004, dev_is_s2,dev_rev_g, dev_type);
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

	delete[] g;
	delete[] rev_g;
	

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
