/* Runs a command and prints its own peak RSS and CPU time (wait4), for measuring the entry on an Android phone,
   where /usr/bin/time does not exist.  measure <command> [args...]  ->  stdout: "peak_rss_kb user_s sys_s exit" */
#include <stdio.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: measure <command> [args...]\n"); return 2; }
    pid_t pid = fork();
    if (pid == 0) { execv(argv[1], argv + 1); perror("execv"); _exit(127); }
    int status = 0;
    struct rusage ru;
    if (wait4(pid, &status, 0, &ru) < 0) { perror("wait4"); return 1; }
    printf("%ld %.6f %.6f %d\n", ru.ru_maxrss, ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6,
           ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return 0;
}
