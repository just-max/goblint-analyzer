#include <pthread.h>
#include <assert.h>

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>


void *foo(void* p){
    sleep(2);
    int* ip = *((int**) p);
    printf("ip is %d\n", *ip);
    *ip = 42;
    return NULL;
}

int main(){
    int x = 0;
    int *xp = &x;
    int** ptr = &xp;
    int x2 = 35;
    pthread_t thread;
    pthread_create(&thread, NULL, foo, ptr);
    assert(x2 == 35);
    *ptr = &x2;
    sleep(4); // to make sure that we actually fail the assert when running.
    assert(x2 == 35); // UNKNOWN!
    pthread_join(thread, NULL);
    return 0;
}
