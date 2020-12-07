#ifndef COMMON_HH
#define COMMON_HH

#include <map>
#include <string_view>
#include <iostream>
#include <chrono>
#include <mpi.h>

#include "cxxopts.hpp"

int mpi_rank, mpi_size, mpi_rank_name_len;
char mpi_rank_name[MPI_MAX_PROCESSOR_NAME], *mpi_rank_names;
std::map<std::string_view, int> node_ranks;

void mpi_init_common(bool ensure_single_rank = true) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // gather names of all ranks
    MPI_Get_processor_name(mpi_rank_name, &mpi_rank_name_len);
    if (mpi_rank == 0) {
        mpi_rank_names = new char[MPI_MAX_PROCESSOR_NAME * mpi_size];
    }
    MPI_Gather(mpi_rank_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, mpi_rank_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        for (int i = 0; i < mpi_size; ++i) {
            auto name = &mpi_rank_names[i * MPI_MAX_PROCESSOR_NAME];
            node_ranks[std::string_view(name)] += 1;
        }
        if (ensure_single_rank){
            for (auto [name, count]: node_ranks) {
                if (count > 1) {
                    fprintf(stderr, "WARNING: multiple (%d) ranks on node %s, results might be incorrect\n", count, name.data());
                }
            }
        }
    }
}


#define DO_RANK_ZERO(EXPR) if (mpi_rank == 0) { EXPR; }
#define EARLY_EXIT(CODE) MPI_Finalize(); return (CODE);

#define MPI_TEST_BATCH_SIZE 20

struct timer {
    template <typename F, typename... T>
    double operator()(F &&func, T... args) {
        auto start = std::chrono::high_resolution_clock::now();
        func(std::forward<T>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        return duration.count();
    };
};

#endif
