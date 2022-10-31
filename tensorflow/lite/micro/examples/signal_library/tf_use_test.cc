
#include <tensorflow/core/util/util.h>

int main(int argc, char* argv[]) {
    const char* ptr = "test";
    size_t n = 4;
    tensorflow::PrintMemory(ptr, n);
    print("here");
}