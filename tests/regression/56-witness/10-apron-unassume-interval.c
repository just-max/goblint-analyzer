// SKIP PARAM: --set ana.activated[+] apron --set ana.activated[+] unassume --set witness.yaml.unassume 10-apron-unassume-interval.yml
#include <assert.h>

int main() {
  int i = 0;
  while (i < 100) {
    i++;
  }
  assert(i == 100);
  return 0;
}
