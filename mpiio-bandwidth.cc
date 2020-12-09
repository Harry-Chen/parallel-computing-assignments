#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include <thread>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>

#include "common.hh"

struct tester_thread_config {
    bool read = true;
    int offset = 0;
};

int main(int argc, char *argv[]) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        DO_RANK_ZERO(fprintf(stderr, "ERROR: MPI does not support multiple threads\n"));
    }

    mpi_init_common();

    // parse command line arguments
    std::string program(argv[0]);
    cxxopts::Options options(program, program + " - MPI IO bandwidth tester");
    options.allow_unrecognised_options();
    options.add_options()
        ("h,help", "print help")
        ("s,size", "test block sizes (in bytes)", cxxopts::value<std::vector<size_t>>()->default_value("1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131972,262144,524288,1048576"))
        ("r,ratio", "ratio of write", cxxopts::value<std::vector<int>>()->default_value("0,30,50,70,100"))
        ("t,thread", "thread number of each rank", cxxopts::value<int>()->default_value("10"))
        ;

    // fetch options
    auto result = options.parse(argc, argv);
    // test sizes (sort + uniquefy)
    auto test_sizes = result["s"].as<std::vector<size_t>>();
    std::sort(test_sizes.begin(), test_sizes.end());
    test_sizes.erase(std::unique(test_sizes.begin(), test_sizes.end()), test_sizes.end());
    // other options
    auto threads = result["t"].as<int>();
    // ratios (sort + uniquefy)
    auto ratios = result["r"].as<std::vector<int>>();
    std::sort(ratios.begin(), ratios.end());
    ratios.erase(std::unique(ratios.begin(), ratios.end()), ratios.end());

    // print help
    if (result.count("h")) {
        DO_RANK_ZERO(std::cerr << options.help({""}) << std::endl);
        EARLY_EXIT(0);
    }

    if (test_sizes.size() < 1) {
        DO_RANK_ZERO(fprintf(stderr, "ERROR: no test size given.\n"));
        EARLY_EXIT(1);
    }

    if (threads < 1) {
        DO_RANK_ZERO(fprintf(stderr, "ERROR: total threads must be >= 1.\n"));
        EARLY_EXIT(1);
    }

    for (auto r : ratios) {
        if (r < 0 || r > 100) {
            DO_RANK_ZERO(fprintf(stderr, "ERROR: write ratio must be in range [0, 100]: %d.\n", r));
            EARLY_EXIT(1);
        }
    }

    for (auto s : test_sizes) {
        if (s <= 0) {
            DO_RANK_ZERO(fprintf(stderr, "ERROR: test size must be > 0: %zu.\n", s));
            EARLY_EXIT(1);
        }
    }

    // print test parameters
    if (mpi_rank == 0) {
        puts("MPI IO bandwidth tester");
        printf("Nodes: ");
        for (int i = 0; i < mpi_size; ++i) {
            printf("%s%s", &mpi_rank_names[i * MPI_MAX_PROCESSOR_NAME], ((i == mpi_size - 1) ? "\n" : ", "));
        }
        printf("Test sizes: ");
        for (size_t i = 0; i < test_sizes.size(); i++) {
            printf("%zu%s", test_sizes[i], ((i == test_sizes.size() - 1) ? "\n" : ", "));
        }
        printf("Test ratios: ");
        for (size_t i = 0; i < ratios.size(); i++) {
            printf("%d%s", ratios[i], ((i == ratios.size() - 1) ? "\n" : ", "));
        }
        printf("Threads on each rank: %d\n", threads);
    }

    auto configs = new tester_thread_config[threads]();
    auto test_threads = new std::thread[threads];
    auto max_size_per_thread = 16 * 1048576UL; // 16 MB
    auto buf = new uint8_t[max_size_per_thread * threads]; // write / read buffer
    auto result_buffer = new double[mpi_size];
    // auto requests = new MPI_Request[threads * 256];

    // iterate over ratios
    for (auto r : ratios) {
        int write_threads = std::round((double) threads * r / 100);
        auto read_threads = threads - write_threads;
        DO_RANK_ZERO(puts("=================="));
        DO_RANK_ZERO(printf("Write ratio: %d%% (%d read threads, %d write threads)\n", r, read_threads, write_threads));

        // print header
        if (mpi_rank == 0) {
            printf("  Size  ");
            for (int p = 0; p < mpi_size; ++p) {
                printf("   %3d    ", p);
            }
            puts("Collective");
        }

        for (int i = 0; i < write_threads; ++i) configs[i].read = false; // first W threads write

        // iterate over sizes
        for (size_t s = 0; s < test_sizes.size(); ++s) {
            auto size = test_sizes[s];

            size_t size_per_thread = std::min(size * 256, max_size_per_thread); // each thread does 256 rounds of IO at maximum
            size_t iter_per_thread = size_per_thread / size; // IO operations performed by each thread
            size_t size_per_rank = threads * size_per_thread;
            size_t round_size = size_per_rank * mpi_size;
            size_t total_file_size = round_size * 2; // non-collective / collective

            // create file for current round of testing
            char test_file_name[] = "io_test_XXXXXX";
            if (mpi_rank == 0) {
                int temp_fd = mkstemp(test_file_name);
                if (temp_fd == -1) {
                    perror("Generate random file name failed");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (ftruncate(temp_fd, total_file_size) != 0) {
                    perror("Create testing file failed");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (close(temp_fd) != 0) {
                    perror("Close testing file failed");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            MPI_Bcast(test_file_name, sizeof(test_file_name), MPI_CHAR, 0, MPI_COMM_WORLD);

            // open test file
            MPI_File file;
            if (MPI_File_open(MPI_COMM_WORLD, test_file_name, MPI_MODE_DELETE_ON_CLOSE | MPI_MODE_RDWR, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
                fprintf(stderr, "ERROR: cannot open test file %s on rank %d\n", test_file_name, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }

            // config offsets
            for (int i = 0; i < threads; ++i) {
                configs[i].offset = mpi_rank * size_per_rank + i * size_per_thread;
            }

            double collective_result = 0.0, my_result = 0.0;

            // test on each rank
            for (int r = 0; r < mpi_size + 1; ++r) {
                // either my round, or collective round
                if (r == mpi_rank || r == mpi_size) {
                    // in the final round, increase offset
                    if (r == mpi_size) {
                        for (int i = 0; i < threads; ++i) {
                            configs[i].offset += round_size;
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                    }
                    // spawn test threads
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < threads; ++i) {
                        test_threads[i] = std::thread([=](){
                            for (size_t j = 0; j < iter_per_thread; ++j) {
                                if (configs[i].read) {
                                    MPI_File_read_at(file, configs[i].offset + j * size, buf + i * max_size_per_thread + j * size, size, MPI_CHAR, MPI_STATUS_IGNORE);
                                } else {
                                    MPI_File_write_at(file, configs[i].offset + j * size, buf + i * max_size_per_thread + j * size, size, MPI_CHAR, MPI_STATUS_IGNORE);
                                }
                            }
                            // MPI_Waitall(iter_per_thread, &requests[256 * i], MPI_STATUSES_IGNORE);
                        });
                    }
                    // wait for all tests to finish
                    for (int i = 0; i < threads; ++i) {
                        test_threads[i].join();
                    }
                    // in the final round, join all ranks
                    if (r == mpi_size) {
                        MPI_Barrier(MPI_COMM_WORLD);
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration<double>(end - start).count();
                    if (r == mpi_size) {
                        collective_result = (double) round_size / 1048576 / duration;
                    } else if (r == mpi_rank) {
                        my_result = (double) size_per_rank / 1048576 / duration;
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            MPI_Gather(&my_result, 1, MPI_DOUBLE, result_buffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // print result line
            if (mpi_rank == 0) {
                printf("%7zu ", size);
                for (int p = 0; p < mpi_size + 1; ++p) {
                    printf("%9.3lf ", p == mpi_size ? collective_result : result_buffer[p]);
                }
                puts("");
            }

            // close test file
            if (MPI_File_close(&file) != MPI_SUCCESS) {
                fprintf(stderr, "ERROR: cannot close test file %s on rank %d\n", test_file_name, mpi_rank);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }
    }

    MPI_Finalize();
    return 0;

}
