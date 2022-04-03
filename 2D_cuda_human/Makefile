common = human2D.cu

CC  	=	nvcc #g++

CFLAGS	=	-w -O3 #-g:warning output to screen   -w:warning ignored

CFLAGS2 =  -arch sm_61 -Xptxas -dlcm=cg

model: $(common)
	$(CC) $(CFLAGS) $(CFLAGS2) -o model $(common)

clean:
	rm model
