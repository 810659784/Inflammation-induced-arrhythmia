#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "Cell.cc"
#include "TerkRatLV.h"

using namespace std;


//#define Inflammation
//#define TNF
//#define IL6
//#define IL1

TerkRatLV::TerkRatLV(CellType ct)
{
	init(ct);
}

void TerkRatLV::init(CellType ct)
{
	Volt = -80.50146;
	Nai = 10.73519;
	Cai = 0.00007901351;
	CaSR = 700.0e-3;
	Ki = 139.2751;
	TRPN = 0.067593139865;
	m = 0.004164108;
	h = 0.6735613;
	j = 0.6729362;
	r = 0.002191519;
	s = 0.9842542;
	sslow = 0.6421196;
	rss = 0.002907171;
	sss = 0.3142767;
	y = 0.003578708;
	z = 0.014417937837;
	Q1 = 0;
	Q2 = 0;
	Q3 = 0;
	z1 = 0.98859;
	z2 = 0.0087302;
	z3 = 0.0026566;

	ctype = ct;
}
void TerkRatLV::update()
{
	double R = 8314.5;
	double T = 295;
	double F = 96487;
	double Cm = 0.0001;
	double Vmyo = 25.85e3;
	double VSR = 2.098e3;
	double VmyouL = 25.85e-6;
	double VSRuL = 2.098e-6;
	double INaKmax = 0.95e-4;
	double KmK = 1.5;
	double KmNa_NaCa_NaK = 10;
	double Ko = 5.4;
	double Nao = 140;
	double CaTRPNMax = 70e-3;
	double gNa = 1.33*0.8;
	if(ctype == LVEPI)
		gNa = 0.8;

	double gt = 0.4647*0.035;

	if(ctype == LVEPI)
		gt = 0.035;
 

   #ifdef Inflammation
   	gt *=0.4;
   #endif

   #ifdef TNF
	gt *= 0.76;
	#endif


	#ifdef IL1
		gt *= 0.63;
	#endif

	double aT0 = 0.583;
	double bT0= 0.417;
	if(ctype == LVEPI)
	{
		aT0 = 0.886;
		bT0 = 0.114;
	}

	double gss = 0.007;
	double gK1 = 0.024;
	double gf = 0.00145;
	double fNa = 0.2;
	double gBNa = 0.00008015;
	double gBCa = 0.0000324;
	double gBK = 0.000138;
	double ECa = 65;
	double Cao = 1.2;
	double gD = 0.065;
	double JR = 0.02;
	double JL = 9.13e-4;
	double N = 50000;
	double KmNa_NaCa = 87.5;
	double KmCa = 1.38;
	double eta = 0.35;
	double ksat = 0.1;
	double gNCX = 38.5e-3;

	double gSERCA = 0.45e-3;
	
	double KSERCA = 0.5e-3;
	double gpCa = 0.0035e-3;
	double KmpCa = 0.5e-3;
	double gCaB = 2.6875e-8;
	double gSRl = 1.8951e-5;
	double kCMDN = 2.382e-3;
	double BCMDN = 50e-3;
	double kon = 100;
	double kRefoff = 0.2;
	double gammatrpn = 2;
	double alpha0 = 8e-3;
	double alphar1 = 2e-3;
	double alphar2 = 1.75e-3;
	double nRel = 3;
	double Kz = 0.15;
	double nHill = 3;
	double Ca50ref = 1.05e-3;
	double zp = 0.85;
	double beta1 = -4;
	double beta0 = 4.9;
	double Tref = 56.2;
	double a = 0.35;
	double A1 = -29;
	double A2 = 138;
	double A3 = 129;
	double alpha1 = 0.03;
	double alpha2 = 0.13;
	double alpha3 = 0.625;
	double VL = -2;
	double delVL = 7;
	double phiL = 2.35;
	double tL = 1;
	double tauL = 650;
	double tauR = 2.43;
	double phiR = 0.05;
	double thetaR = 0.012;
	double KRyR = 41e-3;
	double KL = 0.22e-3;
	double aCT = 0.0625;
	double bCT = 14;
	double cCT = 0.01;
	double dCT = 100;
	double sigma = (exp(Nao/67.3000) - 1.00000)/7.00000;
	//double gNa =  1.33000*gNa;
	double fK = 1.00000 - fNa;
	double K2 =  (( alphar2*pow(zp, nRel))/(pow(zp, nRel)+pow(Kz, nRel)))*(1.00000 - ( nRel*pow(Kz, nRel))/(pow(zp, nRel)+pow(Kz, nRel)));
	double dExtensionRatiodt = 0.00000;
	double tausss = 2.10000;
	double tR =  1.17000*tL;
	double K1 = ( alphar2*pow(zp, nRel - 1.00000)*nRel*pow(Kz, nRel))/pow(pow(zp, nRel)+pow(Kz, nRel), 2.00000);
	double alpham = phiL/tL;
	double betam = phiR/tR;

	double sss_inf,m_inf,taum,h_inf,tauh,mL_inf,taumL,hL_inf,tauhL,j_inf,tauj,taur,r_inf,taus,s_inf,tausslow,sslow_inf,taurss,rss_inf,
			tauy,y_inf,EK,ENa,FVRT,tempINaCa,Cab,ExtensionRatio,lambda,Ca50,CaTRPN50,alphaTm,betaTm,overlap,zmax,TBase,
			T0,Q,Tension,koff,JTRPN,FVRTCa,Coc,mupoc,mupcc,expVL,alphap,betapcc,betapoc,denom,yoc,ycc,r1,mumoc,mumcc,
			r2,epsilonpcc,r7,epsilonm,r8,z4,Cco,epsilonpco,yco,r5,r6,r3,r4,yoo,JRco,JRoo,JR1,JR3,tempIRyR,JLoo,JLoc,
			JL1,JL2,tempILCC,tempIpCa,tempICaB,betaCMDN,Ccc,Coo,lambdaprev,yci,yoi,yic,yio,yii,dQ1dt,dQ2dt,dQ3dt,dsssdt,
			dmdt,dhdt,dmLdt,dhLdt,djdt,drdt,dsdt,dsslowdt,drssdt,dydt,dKidt,dNaidt,dzdt,dTRPNdt,dz2dt,dz1dt,dz3dt,dCaSRdt,dCaidt;

	//...................Ito...............
	EK =  (( R*T)/F)*log(Ko/Ki);
	It =  gt*r*( aT0*s+ bT0* sslow )*(Volt - EK);
	r_inf = 1.00000/(1.00000+exp((Volt+10.6000)/- 11.4200)); //稳态激活

	s_inf = 1.00000/(1.00000+exp((Volt+45.3000)/6.88410));   //稳态失活曲线
	sslow_inf = 1.00000/(1.00000+exp((Volt+45.3000)/6.88410));   //稳态失活曲线

	#ifdef Inflammation
	s_inf = 1.00000/(1.00000+exp(((Volt-5.7)+45.3000)/6.88410));  
	sslow_inf = 1.00000/(1.00000+exp(((Volt-5.7)+45.3000)/6.88410));   
	#endif

	#ifdef TNF
	s_inf = 1.00000/(1.00000+exp(((Volt-5.7)+45.3000)/6.88410));  
	sslow_inf = 1.00000/(1.00000+exp(((Volt-5.7)+45.3000)/6.88410));   
	#endif

	taur = 1.00000/( 45.1600*exp( 0.0357700*(Volt+50.0000))+ 98.9000*exp( - 0.100000*(Volt+38.0000)));
	taus =  0.55*exp(- pow((Volt+70.0)/25.0, 2))+0.049;
	tausslow =  3.3*exp(( (- (Volt+70.0)/30.0)*(Volt+70.0))/30.0)+0.049;
	if(ctype == LVEPI)
	{
		taus = 0.35*exp(-pow((Volt+70.0)/15.0,2))+0.035;
		tausslow = 3.7*exp(-pow((Volt+70.0)/30.0,2))+0.035;
	}
	
	//...................Iss.................................
	Iss =  gss*rss*sss*(Volt - EK);
	rss_inf = 1.00000/(1.00000+exp((Volt+11.5000)/- 11.8200));
	sss_inf = 1.00000/(1.00000+exp((Volt+87.5000)/10.3000));
	taurss = 10.0000/( 45.1600*exp( 0.0357700*(Volt+50.0000))+ 98.9000*exp( - 0.100000*(Volt+38.0000)));

	//...................inward rectifier K+ current....................
	IK1 = ( (48.0000/(exp((Volt+37.0000)/25.0000)+exp((Volt+37.0000)/- 25.0000))+10.0000)*0.00100000)/(1.00000+exp((Volt - (EK+76.7700))/- 17.0000))+( gK1*(Volt - (EK+1.73000)))/( (1.00000+exp(( 1.61300*F*(Volt - (EK+1.73000)))/( R*T)))*(1.00000+exp((Ko - 0.998800)/- 0.124000)));
	IBK =  gBK*(Volt - EK);
	INaK = ( (( (( INaKmax*1.00000)/(1.00000+ 0.124500*exp(( - 0.100000*Volt*F)/( R*T))+ 0.0365000*sigma*exp(( - Volt*F)/( R*T))))*Ko)/(Ko+KmK))*1.00000)/(1.00000+pow(KmNa_NaCa_NaK/Nai, 4.00000));
	IfK =  gf*y*fK*(Volt - EK);
	ENa =  (( R*T)/F)*log(Nao/Nai);
	INa =  gNa*pow(m, 3.00000)*h*j*(Volt - ENa);

	m_inf = 1.00000/(1.00000+exp((Volt+45.0000)/- 6.50000));
	h_inf = 1.00000/(1.00000+exp((Volt+76.1000)/6.07000));
	j_inf = 1.00000/(1.00000+exp((Volt+76.1000)/6.07000));
	y_inf = 1.00000/(1.00000+exp((Volt+138.600)/10.4800));

	taum = 0.00136000/(( 0.320000*(Volt+47.1300))/(1.00000 - exp( - 0.100000*(Volt+47.1300)))+ 0.0800000*exp(- Volt/11.0000));
	tauh = (Volt>=- 40.0000 ?  0.000453700*(1.00000+exp(- (Volt+10.6600)/11.1000)) : 0.00349000/( 0.135000*exp(- (Volt+80.0000)/6.80000)+ 3.56000*exp( 0.0790000*Volt)+ 310000*exp( 0.350000*Volt)));
	tauj = (Volt>=- 40.0000 ? ( 0.0116300*(1.00000+exp( - 0.100000*(Volt+32.0000))))/exp( - 2.53500e-07*Volt) : 0.00349000/( ((Volt+37.7800)/(1.00000+exp( 0.311000*(Volt+79.2300))))*( - 127140*exp( 0.244400*Volt) -  3.47400e-05*exp( - 0.0439100*Volt))+( 0.121200*exp( - 0.0105200*Volt))/(1.00000+exp( - 0.137800*(Volt+40.1400)))));
	tauy = 1.00000/( 0.118850*exp((Volt+80.0000)/28.3700)+ 0.562300*exp((Volt+80.0000)/- 14.1900));

	IBNa =  gBNa*(Volt - ENa);
	FVRT = ( F*Volt)/( R*T);


	//.........Na-Ca exchanger,INaCa
	tempINaCa = ( gNCX*( exp( eta*FVRT)*pow(Nai, 3.00000)*Cao -  exp( (eta - 1.00000)*FVRT)*pow(Nao, 3.00000)*Cai))/( (pow(Nao, 3.00000)+pow(KmNa_NaCa, 3.00000))*(Cao+KmCa)*(1.00000+ ksat*exp( (eta - 1.00000)*FVRT)));
	INaCa =  tempINaCa*VmyouL*F;
	IfNa =  gf*y*fNa*(Volt - ENa);
	Cab = CaTRPNMax - TRPN;
	ExtensionRatio =1;//(VOI>300000. ? 1.00000 : 1.00000);
	lambda = (ExtensionRatio>0.800000&&ExtensionRatio<=1.15000 ? ExtensionRatio : ExtensionRatio>1.15000 ? 1.15000 : 0.800000);
	Ca50 =  Ca50ref*(1.00000+ beta1*(lambda - 1.00000));
	CaTRPN50 = ( Ca50*CaTRPNMax)/(Ca50+ (kRefoff/kon)*(1.00000 - ( (1.00000+ beta0*(lambda - 1.00000))*0.500000)/gammatrpn));
	alphaTm =  alpha0*pow(Cab/CaTRPN50, nHill);
	betaTm = alphar1+( alphar2*pow(z, nRel - 1.00000))/(pow(z, nRel)+pow(Kz, nRel));
	overlap = 1.00000+ beta0*(lambda - 1.00000);
	zmax = (alpha0/pow(CaTRPN50/CaTRPNMax, nHill) - K2)/(alphar1+K1+alpha0/pow(CaTRPN50/CaTRPNMax, nHill));
	TBase = ( Tref*z)/zmax;
	T0 =  TBase*overlap;
	Q = Q1+Q2+Q3;
	Tension = (Q<0.00000 ? ( T0*( a*Q+1.00000))/(1.00000 - Q) : ( T0*(1.00000+ (a+2.00000)*Q))/(1.00000+Q));
	koff = (1.00000 - Tension/( gammatrpn*Tref)>0.100000 ?  kRefoff*(1.00000 - Tension/( gammatrpn*Tref)) :  kRefoff*0.100000);
	JTRPN =  (CaTRPNMax - TRPN)*koff -  Cai*TRPN*kon;
	FVRTCa =  2.00000*FVRT;
	Coc = (fabs(FVRTCa)>1.00000e-09 ? (Cai+( (JL/gD)*Cao*FVRTCa*exp(- FVRTCa))/(1.00000 - exp(- FVRTCa)))/(1.00000+( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa))) : (Cai+ (JL/gD)*Cao)/(1.00000+JL/gD));
	mupoc = (pow(Coc, 2.00000)+ cCT*pow(KRyR, 2.00000))/( tauR*(pow(Coc, 2.00000)+pow(KRyR, 2.00000)));
	mupcc = (pow(Cai, 2.00000)+ cCT*pow(KRyR, 2.00000))/( tauR*(pow(Cai, 2.00000)+pow(KRyR, 2.00000)));
	expVL = exp((Volt - VL)/delVL);
	alphap = expVL/( tL*(expVL+1.00000));
	betapcc = pow(Cai, 2.00000)/( tR*(pow(Cai, 2.00000)+pow(KRyR, 2.00000)));
	betapoc = pow(Coc, 2.00000)/( tR*(pow(Coc, 2.00000)+pow(KRyR, 2.00000)));
	denom =  (alphap+alpham)*( (alpham+betam+betapoc)*(betam+betapcc)+ alphap*(betam+betapoc));
	yoc = ( alphap*betam*(alphap+alpham+betam+betapcc))/denom;
	ycc = ( alpham*betam*(alpham+alphap+betam+betapoc))/denom;
	r1 =  yoc*mupoc+ ycc*mupcc;
	mumoc = ( thetaR*dCT*(pow(Coc, 2.00000)+ cCT*pow(KRyR, 2.00000)))/( tauR*( dCT*pow(Coc, 2.00000)+ cCT*pow(KRyR, 2.00000)));
	mumcc = ( thetaR*dCT*(pow(Cai, 2.00000)+ cCT*pow(KRyR, 2.00000)))/( tauR*( dCT*pow(Cai, 2.00000)+ cCT*pow(KRyR, 2.00000)));
	r2 = ( alphap*mumoc+ alpham*mumcc)/(alphap+alpham);
	epsilonpcc = ( Cai*(expVL+aCT))/( tauL*KL*(expVL+1.00000));
	r7 = ( alpham*epsilonpcc)/(alphap+alpham);
	epsilonm = ( bCT*(expVL+aCT))/( tauL*( bCT*expVL+aCT));
	r8 = epsilonm;
	z4 = ((1.00000 - z1) - z2) - z3;
	Cco = (Cai+ (JR/gD)*CaSR)/(1.00000+JR/gD);
	epsilonpco = ( Cco*(expVL+aCT))/( tauL*KL*(expVL+1.00000));
	yco = ( alpham*( betapcc*(alpham+betam+betapoc)+ betapoc*alphap))/denom;
	r5 =  yco*epsilonpco+ ycc*epsilonpcc;
	r6 = epsilonm;
	r3 = ( betam*mupcc)/(betam+betapcc);
	r4 = mumcc;
	yoo = ( alphap*( betapoc*(alphap+betam+betapcc)+ betapcc*alpham))/denom;
	JRco = ( JR*(CaSR - Cai))/(1.00000+JR/gD);
	JRoo = (fabs(FVRTCa)>1.00000e-05 ? ( JR*((CaSR - Cai)+ (( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa)))*(CaSR -  Cao*exp(- FVRTCa))))/(1.00000+JR/gD+( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa))) : ( JR*((CaSR - Cai)+ (( (JL/gD)*1.00000e-05)/(1.00000 - exp(- 1.00000e-05)))*(CaSR -  Cao*exp(- 1.00000e-05))))/(1.00000+JR/gD+( (JL/gD)*1.00000e-05)/(1.00000 - exp(- 1.00000e-05))));
	JR1 =  yoo*JRoo+ JRco*yco;
	JR3 = ( JRco*betapcc)/(betam+betapcc);
	tempIRyR = ( ( z1*JR1+ z3*JR3)*N)/Vmyo;

	IRyR =  1.50000*tempIRyR;

	ISERCA = ( gSERCA*pow(Cai, 2.00000))/(pow(KSERCA, 2.00000)+pow(Cai, 2.00000)); //Jup
	ISR =  gSRl*(CaSR - Cai); //Jleak


	#ifdef Inflammation
	ISERCA *= 0.8;
	ISR *= 1.63;
	#endif

	#ifdef IL6
		ISERCA *= 0.8;
	#endif

	#ifdef IL1
		ISR *= 1.63;
	#endif
	

	If = IfNa+IfK;
	JLoo = (fabs(FVRTCa)>1.00000e-05 ? ( (( JL*FVRTCa)/(1.00000 - exp(- FVRTCa)))*(( Cao*exp(- FVRTCa) - Cai)+ (JR/gD)*( Cao*exp(- FVRTCa) - CaSR)))/(1.00000+JR/gD+( (JL/gD)*FVRTCa)/(1.00000 - exp(FVRTCa))) : ( (( JL*1.00000e-05)/(1.00000 - exp(- 1.00000e-05)))*(( Cao*exp(- 1.00000e-05) - Cai)+ (JR/gD)*( Cao*exp(- 1.00000e-05) - CaSR)))/(1.00000+JR/gD+( (JL/gD)*1.00000e-05)/(1.00000 - exp(- 1.00000e-05))));
	JLoc = (fabs(FVRTCa)>1.00000e-05 ? ( (( JL*FVRTCa)/(1.00000 - exp(- FVRTCa)))*( Cao*exp(- FVRTCa) - Cai))/(1.00000+( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa))) : ( (( JL*1.00000e-05)/(1.00000 - exp(- 1.00000e-05)))*( Cao*exp(- 1.00000e-05) - Cai))/(1.00000+( (JL/gD)*1.00000e-05)/(1.00000 - exp(- 1.00000e-05))));
	JL1 =  JLoo*yoo+ JLoc*yoc;
	JL2 = ( JLoc*alphap)/(alphap+alpham);
	tempILCC = ( ( z1*JL1+ z2*JL2)*N)/Vmyo;
	ILCC =  - 1.50000*tempILCC*2.00000*VmyouL*F;
	ILCC *= 0.48;// edit 

	#ifdef Inflammation
	ILCC *= 1.27;
	#endif

	#ifdef IL6
		ILCC *= 1.27;
	#endif
   
   

	tempIpCa = ( gpCa*Cai)/(KmpCa+Cai);
	IpCa =  tempIpCa*2.00000*VmyouL*F;
	ECa =  (( R*T)/( 2.00000*F))*log(Cao/Cai);
	tempICaB =  gCaB*(ECa - Volt);
	ICaB =  - tempICaB*2.00000*VmyouL*F;
	betaCMDN = pow(1.00000+( kCMDN*BCMDN)/pow(kCMDN+Cai, 2.00000), - 1.00000);

	Ccc = Cai;
	Coo = (fabs(FVRTCa)>1.00000e-09 ? (Cai+ (JR/gD)*CaSR+( (JL/gD)*Cao*FVRTCa*exp(- FVRTCa))/(1.00000 - exp(- FVRTCa)))/(1.00000+JR/gD+( (JL/gD)*FVRTCa)/(1.00000 - exp(- FVRTCa))) : (Cai+ (JR/gD)*CaSR+ (JL/gD)*Cao)/(1.00000+JR/gD+JL/gD));
	IBCa =  gBCa*(Volt - ECa);
	IB = IBNa+IBCa+IBK;
	lambdaprev = ExtensionRatio;
	yci = alpham/(alphap+alpham);
	yoi = alphap/(alphap+alpham);
	yic = betam/(betapcc+betam);
	yio = betapcc/(betapcc+betam);
	yii = (((((((1.00000 - yoc) - yco) - yoo) - ycc) - yci) - yic) - yoi) - yio;

	//........................update d /dt...........................
	dQ1dt =  A1*dExtensionRatiodt -  alpha1*Q1;
	dQ2dt =  A2*dExtensionRatiodt -  alpha2*Q2;
	dQ3dt =  A3*dExtensionRatiodt -  alpha3*Q3;
	dsssdt = ((sss_inf - sss)/tausss)*0.00100000;
	dmdt = ((m_inf - m)/taum)*0.00100000;
	dhdt = ((h_inf - h)/tauh)*0.00100000;
	djdt = ((j_inf - j)/tauj)*0.00100000;
	drdt = ((r_inf - r)/taur)*0.00100000;
	dsdt = ((s_inf - s)/taus)*0.00100000;
	dsslowdt = ((sslow_inf - sslow)/tausslow)*0.001;
	drssdt = ((rss_inf - rss)/taurss)*0.00100000;
	dydt = ((y_inf - y)/tauy)*0.00100000;
	dKidt = ( - (Istim+Iss*0.00100000+IBK*0.00100000+It*0.00100000+IK1*0.00100000+IfK*0.00100000+ INaK*- 2.00000)*1.00000)/( VmyouL*F);
	dNaidt = ( - (INa*0.00100000+IBNa*0.00100000+ INaCa*3.00000+ INaK*3.00000+IfNa*0.00100000)*1.00000)/( VmyouL*F);
	dzdt =  alphaTm*(1.00000 - z) -  betaTm*z;
	dTRPNdt = JTRPN;
	dz2dt = ( r1*z1 -  (r2+r7)*z2)+ r8*z4;
	dz1dt =  - (r1+r5)*z1+ r2*z2+ r6*z3;
	dz3dt = ( r5*z1 -  (r6+r3)*z3)+ r4*z4;
	dCaSRdt =  (VmyouL/VSRuL)*((- IRyR+ISERCA) - ISR);
	dvdt = - (INa*0.00100000+It*0.00100000+Iss*0.00100000+If*0.00100000+IK1*0.00100000+IBNa*0.00100000+IBK*0.00100000+INaK+Istim+ICaB+INaCa+IpCa+ILCC)/Cm;
	dCaidt =  betaCMDN*(((IRyR - ISERCA)+ISR+JTRPN) - ( - 2.00000*INaCa+IpCa+ICaB+ILCC)/( 2.00000*VmyouL*F));


	// ----------------update states----------------
	Volt += dt*(dvdt+dvgapdt);// is d/dt V in component membrane (mV).
 	Nai += dt*dNaidt;// is d/dt Na_i in component intracellular_ion_concentrations (mM).
 	Ki += dt*dKidt;// is d/dt K_i in component intracellular_ion_concentrations (mM).
 	TRPN += dt*dTRPNdt;// is d/dt TRPN in component intracellular_ion_concentrations (mM).
 	Cai += dt*dCaidt;// is d/dt Ca_i in component intracellular_ion_concentrations (mM).
 	CaSR += dt*dCaSRdt;// is d/dt Ca_SR in component intracellular_ion_concentrations (mM).
 	z += dt*dzdt;// is d/dt z in component tropomyosin (dimensionless).
 	Q1 += dt*dQ1dt;// is d/dt Q_1 in component Cross_Bridges (dimensionless).
 	Q2 += dt*dQ2dt;// is d/dt Q_2 in component Cross_Bridges (dimensionless).
 	Q3 += dt*dQ3dt;// is d/dt Q_3 in component Cross_Bridges (dimensionless).
 	m += dt*dmdt;// is d/dt m in component sodium_current_m_gate (dimensionless).
 	h += dt*dhdt;// is d/dt h in component sodium_current_h_gate (dimensionless).
 	j += dt*djdt;// is d/dt j in component sodium_current_j_gate (dimensionless).
 	r += dt*drdt;// is d/dt r in component Ca_independent_transient_outward_K_current_r_gate (dimensionless).
 	s += dt*dsdt;// is d/dt s in component Ca_independent_transient_outward_K_current_s_gate (dimensionless).
 	sslow += dt*dsslowdt;// is d/dt s_slow in component Ca_independent_transient_outward_K_current_s_slow_gate (dimensionless).
 	rss += dt*drssdt;// is d/dt r_ss in component steady_state_outward_K_current_r_ss_gate (dimensionless).
 	sss += dt*dsssdt;// is d/dt s_ss in component steady_state_outward_K_current_s_ss_gate (dimensionless).
 	y += dt*dydt;// is d/dt y in component hyperpolarisation_activated_current_y_gate (dimensionless).
 	z1 += dt*dz1dt;// is d/dt z_1 in component CaRU_reduced_states (dimensionless).
 	z2 += dt*dz2dt;// is d/dt z_2 in component CaRU_reduced_states (dimensionless).
 	z3 += dt*dz3dt;// is d/dt z_3 in component CaRU_reduced_states (dimensionless).

}


// get & set
double TerkRatLV::getV() {return Volt;}
double TerkRatLV::getNai() {return Nai;}
double TerkRatLV::getKi()  {return Ki;}
double TerkRatLV::getCai() {return Cai;}
double TerkRatLV::getCaSR() {return CaSR;}


double TerkRatLV::getTRPN() {return       TRPN      ;   }
double TerkRatLV::getm()    {return         m       ;   }
double TerkRatLV::geth()    {return         h       ;   }
double TerkRatLV::getj()    {return          j      ;   }
double TerkRatLV::getr()    {return          r      ;   }
double TerkRatLV::gets()    {return          s      ;   }
double TerkRatLV::getsslow(){return           sslow ;   }
double TerkRatLV::getrss()  {return          rss    ;   }
double TerkRatLV::getsss()  {return            sss  ;   }
double TerkRatLV::gety()    {return         y       ;   }
double TerkRatLV::getz()    {return        z        ;   }
double TerkRatLV::getQ1()   {return        Q1       ;   }
double TerkRatLV::getQ2()   {return         Q2      ;   }
double TerkRatLV::getQ3()   {return          Q3     ;   }
double TerkRatLV::getz1()   {return          z1     ;   }
double TerkRatLV::getz2()   {return           z2    ;   }
double TerkRatLV::getz3()   {return           z3    ;   }





double TerkRatLV::getDvdt() {return dvdt;} // unit: mV/ms
double TerkRatLV::getAbsDvdt() {return abs(dvdt);} // unit: mV/ms
double TerkRatLV::getDVgap_dt() {return dvgapdt;} // unit: mV/ms
double TerkRatLV::getINa() {return 1000.0*INa/100;} // unit: pA/pF. i.e. INa/Cm
double TerkRatLV::getIt() {return 1000.0*It/100;}
double TerkRatLV::getIK1() {return 1000.0*IK1/100;} // unit: pA/pF
double TerkRatLV::getICaL() {return 1.e6*ILCC/100;} // unit: pA/pF
double TerkRatLV::getIss() {return 1000.0*Iss/100;}
// double TerkRatLV::getENa() {return 1000.0*Iss/100;}
double TerkRatLV::getIBNa() {return IBNa;}
double TerkRatLV::getINaCa() {return INaCa;}
double TerkRatLV::getIpCa() {return IpCa;}
double TerkRatLV::getIBK() {return IBK;}
double TerkRatLV::getINaK() {return INaK;}
double TerkRatLV::getIfK() {return IfK;}
double TerkRatLV::getIfNa(){return IfNa;}
double TerkRatLV::getIRyR(){return IRyR;}

double TerkRatLV::getISERCA() {return ISERCA;}
double TerkRatLV::getISR() {return ISR;}

double TerkRatLV::getIf() {return If;}
CellType TerkRatLV::getCellType() {return ctype;}
double TerkRatLV::getENa() 
{
	double R = 8314.5;
	double T = 295;
	double F = 96487;
	return (( R*T)/F)*log(140.0/Nai);	
}


void TerkRatLV::setVolt(double param) {Volt = param;} // unit: mV
void TerkRatLV::setIstim(double param) {Istim = param*100.0/1.e6; } // unit: μA.   i.e. Istim*Cm/1e6
void TerkRatLV::setDt(double param) {dt = param;} // unit: ms 
void TerkRatLV::setDVgap_dt(double param) {dvgapdt = param;} // unit: mV/ms
//void TerkRatLV::setCORM2(double param) {CORM2 = param;} // unit: mV/ms



void TerkRatLV::outputAllStates(FILE *datafile)
{
	fprintf(datafile,"%4.10f\n", Volt    ); // = -80.50146;	
	fprintf(datafile,"%4.10f\n", Nai     ); // = 10.73519;
	fprintf(datafile,"%4.10f\n", Cai     ); // = 0.00007901351;
	fprintf(datafile,"%4.10f\n", CaSR    ); // = 700.0e-3;
	fprintf(datafile,"%4.10f\n", Ki      ); // = 139.2751;
	fprintf(datafile,"%4.10f\n", TRPN    ); // = 0.067593139865;
	fprintf(datafile,"%4.10f\n", m       ); // = 0.004164108;
	fprintf(datafile,"%4.10f\n", h       ); // = 0.6735613;
	fprintf(datafile,"%4.10f\n", j       ); // = 0.6729362;
	fprintf(datafile,"%4.10f\n", r       ); // = 0.002191519;
	fprintf(datafile,"%4.10f\n", s       ); // = 0.9842542;
	fprintf(datafile,"%4.10f\n", sslow   ); // = 0.6421196;	
	fprintf(datafile,"%4.10f\n", rss     ); // = 0.002907171;
	fprintf(datafile,"%4.10f\n", sss     ); // = 0.3142767;
	fprintf(datafile,"%4.10f\n", y       ); // = 0.003578708;
	fprintf(datafile,"%4.10f\n", z       ); // = 0.014417937837;	
	fprintf(datafile,"%4.10f\n", Q1      ); // = 0;
	fprintf(datafile,"%4.10f\n", Q2      ); // = 0;
	fprintf(datafile,"%4.10f\n", Q3      ); // = 0;
	fprintf(datafile,"%4.10f\n", z1      ); // = 0.98859;
	fprintf(datafile,"%4.10f\n", z2      ); // = 0.0087302;
	fprintf(datafile,"%4.10f\n", z3      ); // = 0.0026566;
}

void TerkRatLV::readinAllStates(FILE *datafile)
{
	double value;
	fscanf(datafile,"%lf", &value );
	Volt = value;
	fscanf(datafile,"%lf", &value );
	Nai  = value;
	fscanf(datafile,"%lf", &value );
	Cai  = value;
	fscanf(datafile,"%lf", &value );
	CaSR = value;
	fscanf(datafile,"%lf", &value );
	Ki   = value;
	fscanf(datafile,"%lf", &value );
	TRPN = value;
	fscanf(datafile,"%lf", &value );
	m    = value;
	fscanf(datafile,"%lf", &value );
	h    = value;
	fscanf(datafile,"%lf", &value );
	j    = value;
	fscanf(datafile,"%lf", &value );
	r    = value;
	fscanf(datafile,"%lf", &value );
	s    = value;
	fscanf(datafile,"%lf", &value );
	sslow= value;
	fscanf(datafile,"%lf", &value );
	rss  = value;
	fscanf(datafile,"%lf", &value );
	sss  = value;
	fscanf(datafile,"%lf", &value );
	y    = value;
	fscanf(datafile,"%lf", &value );
	z    = value;
	fscanf(datafile,"%lf", &value );
	Q1   = value;
	fscanf(datafile,"%lf", &value );
	Q2   = value;
	fscanf(datafile,"%lf", &value );
	Q3   = value;
	fscanf(datafile,"%lf", &value );
	z1   = value;
	fscanf(datafile,"%lf", &value );
	z2   = value;
	fscanf(datafile,"%lf", &value );
	z3   = value;

}







