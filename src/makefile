SOURCEDIR = .
EXE = DNN

SOURCES = $(wildcard $(SOURCEDIR)/*.cu)
OBJECTS = $(SOURCES:.cu=.o)

CC = nvcc

# Builds SASS for all architectures into one executable
ARCHS = -arch=sm_75 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_37,code=sm_37 \

LIBS=""

#
## Debug build settings
#
DBGDIR = debug
DBGEXE = $(DBGDIR)/$(EXE)
DBGOBJS = $(addprefix $(DBGDIR)/, $(OBJECTS))
# rdynamic and lineinfo for running memcheck
DBGFLAGS = -std=c++14 -g -G -Xcompiler -fopenmp -lcuda -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -Xcompiler -rdynamic -DDEBUG

#
## Release build settings
#
RELDIR = release
RELEXE = $(RELDIR)/$(EXE)
RELOBJS = $(addprefix $(RELDIR)/, $(OBJECTS))
RELFLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES

.PHONY: all clean debug prep release remake

# General rules

all: prep release
remake: clean all

prep:
	@mkdir -p $(DBGDIR) $(RELDIR)

clean:
	rm -f $(RELEXE) $(RELOBJS) $(DBGEXE) $(DBGOBJS)

# Debug rules

debug: $(DBGEXE)

$(DBGEXE): $(DBGOBJS)
	$(CC) $(DBGFLAGS) $(ARCHS) $^ -o $@

$(DBGDIR)/%.o: %.cu
	$(CC) $(DBGFLAGS) $(ARCHS) $< -c -o $@

# Release rules

release: $(RELEXE)

$(RELEXE): $(RELOBJS)
	$(CC) $(RELFLAGS) $(ARCHS) $^ -o $@

$(RELDIR)/%.o: %.cu
	$(CC) $(RELFLAGS) $(ARCHS) $< -c -o $@
