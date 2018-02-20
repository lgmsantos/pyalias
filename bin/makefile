VPATH=../src/cython
.PHONY: clean

alias.so: alias.o
	$(LD) $(LDFLAGS) -shared $< $(LDLIBS) -o $@

alias.o: CFLAGS+= -fPIC -I/usr/include/python3.5

alias.c: alias.pyx
	cython3 $< -o $@

clean:
	rm -rf *.o *.so alias.c
