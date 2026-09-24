CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Iinclude -g
SRC = src/util.c src/tokenizer.c src/graph.c src/importance.c src/ctm.c src/noise.c src/ivm.c src/inference.c src/interpret.c src/model.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean test

all: bin/mse_train bin/mse_ctm_test bin/mse_ivm_test bin/mse_inference_test bin/mse_interpret_test bin/mse_model_test bin/mse_chat bin/mse_incremental_test bin/mse_train_corpus

bin/mse_train: tools/mse_train.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/mse_ctm_test: tools/mse_ctm_test.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/mse_ivm_test: tools/mse_ivm_test.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/mse_inference_test: tools/mse_inference_test.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/mse_interpret_test: tools/mse_interpret_test.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/mse_model_test: tools/mse_model_test.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/mse_chat: tools/mse_chat.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/mse_incremental_test: tools/mse_incremental_test.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/mse_train_corpus: tools/mse_train_corpus.c $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin:
	mkdir -p bin

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) bin/mse_train bin/mse_ctm_test bin/mse_ivm_test bin/mse_inference_test bin/mse_interpret_test bin/mse_model_test bin/mse_chat bin/mse_incremental_test bin/mse_train_corpus

test: all
	./bin/mse_train sample.txt /tmp/mse_model 500
