#include <iostream>
#include <vector>
#include <string>

#include "common.hh"

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    mpi_init_common();

    // parse command line arguments
    std::string program(argv[0]);
    cxxopts::Options options(program, program + " - MPI P2P bandwidth tester");
    options.allow_unrecognised_options();
    options.add_options()
        ("h,help", "print help")
        ("s,size", "test sizes (in bytes)", cxxopts::value<std::vector<int>>()->default_value("1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131972,262144,524288,1048576,2097152,4194304"))
        ("r,repeat", "repeat times of each test", cxxopts::value<int>()->default_value("8"))
        ("w,warmup", "warmup run numbers of each test", cxxopts::value<int>()->default_value("3"))
        ("b,batch", "send/recv batch size of each test", cxxopts::value<int>()->default_value("20"))
        ("o,output", "output result file", cxxopts::value<std::string>()->default_value(""));

    // fetch options
    auto result = options.parse(argc, argv);
    auto repeat = result["r"].as<int>();
    auto test_sizes = result["s"].as<std::vector<int>>();
    auto warmup = result["w"].as<int>();
    auto batch = result["b"].as<int>();
    std::sort(test_sizes.begin(), test_sizes.end());
    auto output = result["o"].as<std::string>();

    // print help
    if (result.count("h")) {
        DO_RANK_ZERO(std::cerr << options.help({""}) << std::endl);
        EARLY_EXIT(0);
    }

    if (mpi_size <= 1) {
        std::cerr << "ERROR: please run with at least two ranks." << std::endl;
        EARLY_EXIT(1);
    }

    if (test_sizes.size() < 1) {
        DO_RANK_ZERO(fprintf(stderr, "ERROR: no test size given.\n"));
        EARLY_EXIT(1);
    }

    if (repeat < 1) {
        DO_RANK_ZERO(fprintf(stderr, "ERROR: repeat must be > 1.\n"));
        EARLY_EXIT(1);
    }

    if (warmup < 0) {
        DO_RANK_ZERO(fprintf(stderr, "ERROR: warmup run must be >= 0.\n"));
        EARLY_EXIT(1);
    }

    if (batch < 1) {
        DO_RANK_ZERO(fprintf(stderr, "ERROR: batch size must be >= 1.\n"));
        EARLY_EXIT(1);
    }

    // print test parameters
    if (mpi_rank == 0) {
        puts("MPI P2P bandwidth tester");
        printf("Nodes: ");
        for (int i = 0; i < mpi_size; ++i) {
            printf("%s%s", &mpi_rank_names[i * MPI_MAX_PROCESSOR_NAME], ((i == mpi_size - 1) ? "\n" : ", "));
        }
        printf("Repeat times: %d\n", repeat);
        printf("Result file: ");
        if (output != "") {
            printf("%s\n", output.c_str());
        } else {
            puts("[DISABLED]");
        }
        printf("Test sizes: ");
        for (size_t i = 0; i < test_sizes.size(); i++) {
            printf("%d%s", test_sizes[i], ((i == test_sizes.size() - 1) ? "\n" : ", "));
        }
    }

    // open result file
    MPI_File output_file = MPI_FILE_NULL;
    if (output != "") {
        if (MPI_File_open(MPI_COMM_WORLD, output.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &output_file) != MPI_SUCCESS) {
            DO_RANK_ZERO(fprintf(stderr, "ERROR: cannot open output file %s\n", output.c_str()));
            EARLY_EXIT(2);
        };
    }


    size_t max_size = test_sizes[test_sizes.size() - 1];
    auto test_result = new double[mpi_size]();
    double *all_result = nullptr;
    MPI_Request *requests = new MPI_Request[batch];
    auto buf = new uint8_t[max_size * batch]();

    if (mpi_rank == 0) {
        all_result = new double[mpi_size * mpi_size]();
    }

    // running for different sizes
    for (size_t s = 0; s < test_sizes.size(); ++s) {
        int size = test_sizes[s];
        
        // running through N anti-diagonals
        for (int j = -2; j < mpi_size * (warmup + 1); ++j) {
            int i = std::max(0, j) / (warmup + 1); // first W - 1 run is warming up
            int lower_diagonal_sum = mpi_size - 1 + i;
            int upper_diagonal_sum = i - 1;
            int opposite_rank = -1;
            if (mpi_rank <= lower_diagonal_sum && mpi_rank > upper_diagonal_sum) {
                opposite_rank = lower_diagonal_sum - mpi_rank;
            } else if (mpi_rank <= upper_diagonal_sum) {
                opposite_rank = upper_diagonal_sum - mpi_rank;
            }
            timer timer;
            auto transfer_data = [&](int times, bool reverse = false) {
                bool to_send = mpi_rank < opposite_rank;
                bool to_receive = mpi_rank > opposite_rank;
                // reverse send and receive
                if (reverse) {
                    std::swap(to_receive, to_send);
                }
                for (int r = 0; r < times; r++) {
                    // printf("%d sends to %d\n", mpi_rank, opposite_rank);
                    if (to_send) {
                        for (int j = 0; j < batch; ++j) {
                            MPI_Isend(buf + j * max_size, size, MPI_CHAR, opposite_rank, j, MPI_COMM_WORLD, &requests[j]);
                        }
                    } else if (to_receive) {
                        // printf("%d receives from %d\n", mpi_rank, opposite_rank);
                        for (int j = 0; j < batch; ++j) {
                            MPI_Irecv(buf + j * max_size, size, MPI_CHAR, opposite_rank, j, MPI_COMM_WORLD, &requests[j]);
                        }
                    }
                    if (mpi_rank != opposite_rank) {
                        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
                    }
                }
            };

            MPI_Barrier(MPI_COMM_WORLD);
            // smaller rank -> larger rank (receive bandwidth of larger rank)
            // transfer_data(2, false); // warm up
            auto duration_1 = timer(transfer_data, repeat, false);
            MPI_Barrier(MPI_COMM_WORLD);
            // smaller rank <- larger rank (receive bandwidth of smaller rank)
            // transfer_data(2, true); // warm up
            auto duration_2 = timer(transfer_data, repeat, true);
            MPI_Barrier(MPI_COMM_WORLD);

            if (mpi_rank < opposite_rank) {
                test_result[opposite_rank] = (double) size * repeat * batch / 1000000 / duration_2;
            } else if (mpi_rank > opposite_rank) {
                test_result[opposite_rank] = (double) size * repeat * batch / 1000000 / duration_1;
            }
        }

        // gather results to root rank
        MPI_Gather(test_result, mpi_size, MPI_DOUBLE, all_result, mpi_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            puts("==================");
            printf("Send size: %d B\n", size);
            printf("S/R ");
            for (int p = 0; p < mpi_size; ++p) {
                printf("   %3d       ", p);
            }
            puts("");
            for (int p = 0; p < mpi_size; ++p) {
                printf("%3d ", p);
                for (int q = 0; q < mpi_size; ++q) {
                    printf(p == q ? "     NA      " : "%12.6lf ", all_result[p * mpi_size + q]);
                }
                puts("");
            }
        }

        if (output_file != MPI_FILE_NULL) {
            int line_size = sizeof(double) * mpi_size;
            int offset = line_size * mpi_size * s + line_size * mpi_rank;
            MPI_File_write_at_all(output_file, offset, test_result, mpi_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
        }
    }

    if (output_file != MPI_FILE_NULL) {
        MPI_File_close(&output_file);
    }

    MPI_Finalize();
    return 0;
}
