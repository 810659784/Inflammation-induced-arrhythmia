#ifndef _TERKRATLV_H_
#define _TERKRATLV_H_
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <typeinfo>
#include "omp.h"
#include <ctime>
#include "Cell.cc"

using namespace std;



class TerkRatLV: public Cell
{

private:

	CellType ctype;

	// ------------------states------------------
	double dt;
	double Volt ;	
	double Nai ;
	double Cai ;
	double CaSR ;
	double Ki ;
	double TRPN ;
	double m ;
	double h ;
	double mL ;// INaL related.
	double hL ;// INaL related.
	double j ;
	double r ;
	double s ;
	double sslow ;	
	double rss ;
	double sss ;
	double y ;
	double z ;	
	double Q1 ;
	double Q2 ;
	double Q3 ;
	double z1 ;
	double z2 ;
	double z3 ;

	// ------------------currents------------------
	double Istim;
	double It;
	double Iss;
	double IK1;
	double IBK;
	double INaK;
	double IfK;
	double INa;
	double IBNa;
	double INaCa;
	double IfNa;
	double IRyR;
	double ISERCA;
	double ISR;
	double If;
	double ILCC;
	double IpCa;
	double ICaB;
	double IBCa;
	double IB;
	double INaL; // INaL related.

	// others
	double dvdt;
	double dvgapdt;
	double CORM2;






public:
	TerkRatLV(CellType ct);
	void init(CellType ct);
	void update();

	void setVolt(double param) ; // unit: mV
	void setIstim(double param) ;//Istim = param/10.0;}
	void setDt(double param) ;//dt = param/1000;}
	void setDVgap_dt(double param) ;
	// void setDVgap_dt(double param) {dvgapdt = param*1000.0;}// convert to (mV/s)! 
	void setCORM2(double param);


	void outputAllStates(FILE *datafile);
	void readinAllStates(FILE *datafile);

	double getV();
	double getNai() ;
	double getKi()  ;
	double getCai() ;
	double getCaSR();


	double getTRPN() ;
	double getm()    ;
	double geth()    ;
	//double getmL()   ;
	//double gethL()   ;
	double getj()    ;
	double getr()    ;
	double gets()    ;
	double getsslow();
	double getrss()  ;
	double getsss()  ;
	double gety()    ;
	double getz()    ;
	double getQ1()   ;
	double getQ2()   ;
	double getQ3()   ;
	double getz1()   ;
	double getz2()   ;
	double getz3()   ;


	double getDvdt()    ;
	double getAbsDvdt() ;
	double getINa()     ;
	//double getINaL()    ;
	double getIt()      ;
	double getIK1()     ;
	double getICaL()    ;
	double getIss()     ;
	double getIBNa()    ;
	double getINaCa()   ;
	double getIpCa()    ;
	double getIBK()     ;
	double getINaK()    ;
	double getIfK()     ;
	double getIfNa()    ;
	double getIRyR()    ;
	double getISERCA()  ;
	double getISR()     ;
	double getIf()      ;
	double getENa()     ;
	double getDVgap_dt() ; // convert back to mV/ms
	CellType getCellType();

};
#endif