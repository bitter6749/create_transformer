CC = gcc
CFALGS = -Wall -Wextra -pedantic # src/* -Iinclude

main: main.c
	$(CC) main.c -o main $(CFALGS)

run: main
	./main