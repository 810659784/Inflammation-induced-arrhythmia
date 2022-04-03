#ifndef _CELL_CC_
#define _CELL_CC_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

using namespace std;

enum CellType { LVEPI, LVENDO, EPI, ENDO, MCELL,NormalCell,TnfendoCell,TnfepiCell};

class Cell
{
public:
	virtual void init() {}
	virtual void init(CellType ct) {}
	virtual void setIstim(double param) {}
	virtual void setDt(double param) {}
	virtual void setDVgap_dt(double param) {}
	virtual CellType getCellType() {}
	virtual double getDVgap_dt() {}
	virtual double getV() {}
	virtual double getCai() {}

	virtual double getNai()  {};
	virtual double getKi()   {};
	virtual double getCaSR() {};
	virtual double getTRPN() {};
	virtual double getm()    {};
	virtual double geth()    {};
	virtual double getmL()   {};
	virtual double gethL()   {};
	virtual double getj()    {};
	virtual double getr()    {};
	virtual double gets()    {};
	virtual double getsslow(){};
	virtual double getrss()  {};
	virtual double getsss()  {};
	virtual double gety()    {};
	virtual double getz()    {};
	virtual double getQ1()   {};
	virtual double getQ2()   {};
	virtual double getQ3()   {};
	virtual double getz1()   {};
	virtual double getz2()   {};
	virtual double getz3()   {};


	virtual double getIstim() {}
	virtual double getINa(){}
	virtual double getINaL(){}
	virtual double getIt(){}
	virtual double getDvdt() {}
	virtual void update() {}
	virtual void outputAllStates(FILE *datafile) {}
	virtual void readinAllStates(FILE *datafile) {}
};
#endif