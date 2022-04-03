all: SingleCell

common = TP06.cc Cell_TP06.cc   #TP06

CC  	=	g++

CFLAGS	=	-w -O3 #-g:warning output to screen   -w:warning ignored

CFLAGS2	=	-fopenmp


SingleCell: $(common) SingleCell.cc
	$(CC) $(CFLAGS) -o single_cell $(common) SingleCell.cc

ERP: $(common) ERP.cc
	$(CC) $(CFLAGS) -o ERP $(common) ERP.cc

OneD: $(common) OneD.cc
	$(CC) $(CFLAGS) $(CFLAGS2) -o oned $(common) OneD.cc
	
TwoD: $(common) TwoD.cc
	$(CC) $(CFLAGS) $(CFLAGS2) -o twod $(common) TwoD.cc

ecg_TwoD: $(common) TwoD_ecg.cc
	$(CC) $(CFLAGS) $(CFLAGS2) -o ecg_twod $(common) TwoD_ecg.cc

vw: $(common) OneD_VW.cc
	$(CC) $(CFLAGS) $(CFLAGS2) -o vw $(common) OneD_VW.cc


VoltageClamp: $(common) VoltageClampIto.cc
	$(CC) $(CFLAGS) -o VoltageClamp $(common) VoltageClampIto.cc

clean:
	rm model_*
