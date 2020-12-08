#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <limits>

#include "common.hh"


int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    mpi_init_common();

    // parse command line arguments
    std::string program(argv[0]);
    cxxopts::Options options(program, program + " - MPI collective bandwidth tester");
    options.allow_unrecognised_options();
    options.add_options()
        ("h,help", "print help")
        ("s,size", "test sizes (in bytes)", cxxopts::value<std::vector<int>>()->default_value("4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131972,262144,524288,1048576,2097152,4194304"))
        ("m,methods", "collective methods to test", cxxopts::value<std::vector<std::string>>()->default_value("bcast,reduce,gather,scatter,scan,reduce_scatter,allgather,allreduce,alltoall"))
        ("r,repeat", "repeat times of each test", cxxopts::value<int>()->default_value("8"))
        ("w,warmup", "warmup run numbers of each test", cxxopts::value<int>()->default_value("3"))
        ("b,batch", "operation batch size of each test", cxxopts::value<int>()->default_value("20"))
        ("R,roots", "selected roots for single-rooted operation", cxxopts::value<std::vector<int>>()->default_value("0"));

    // fetch options
    auto result = options.parse(argc, argv);
    auto repeat = result["r"].as<int>();
    // test sizes (sort + uniquefy)
    auto test_sizes = result["s"].as<std::vector<int>>();
    std::sort(test_sizes.begin(), test_sizes.end());
    test_sizes.erase(std::unique(test_sizes.begin(), test_sizes.end()), test_sizes.end());
    // methods (sort + uniquefy)
    auto methods = result["m"].as<std::vector<std::string>>();
    std::for_each(methods.begin(), methods.end(), [](std::string s) {std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });}); // convert method names to lower case
    std::sort(methods.begin(), methods.end());
    methods.erase(std::unique(methods.begin(), methods.end()), methods.end());
    // other options
    auto warmup = result["w"].as<int>();
    auto batch = result["b"].as<int>();
    // root nodes (sort + uniquefy)
    auto roots = result["R"].as<std::vector<int>>();
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

    const std::vector<std::string> SINGLE_ROOTED_METHODS = {"bcast", "reduce", "gather", "scatter"};
    const std::vector<std::string> ALL_COLLECTIVE_METHODS = {"scan", "reduce_scatter", "allgather", "allreduce", "alltoall"};

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

    auto is_single_rooted = [&](const std::string &m) { return std::find(SINGLE_ROOTED_METHODS.begin(), SINGLE_ROOTED_METHODS.end(), m) != SINGLE_ROOTED_METHODS.end(); };
    auto is_all_collective = [&](const std::string &m) { return std::find(ALL_COLLECTIVE_METHODS.begin(), ALL_COLLECTIVE_METHODS.end(), m) != ALL_COLLECTIVE_METHODS.end(); };

    for (const auto &m : methods) {
        if (!is_single_rooted(m) && !is_all_collective(m)) {
            DO_RANK_ZERO(fprintf(stderr, "ERROR: invalid method: %s.\n", m.c_str()));
            EARLY_EXIT(1);
        }
    }

    for (auto r : roots) {
        if (r < 0 || r >= mpi_size) {
            DO_RANK_ZERO(fprintf(stderr, "ERROR: invalid rank number of root: %d.\n", r));
            EARLY_EXIT(1);
        }
    }

    for (auto s : test_sizes) {
        if (s <= 0 || s % 4 != 0) {
            DO_RANK_ZERO(fprintf(stderr, "ERROR: test size must be a multiple of 4: %d.\n", s));
            EARLY_EXIT(1);
        }
    }

    // print test parameters
    if (mpi_rank == 0) {
        puts("MPI P2P collective tester");
        printf("Nodes: ");
        for (int i = 0; i < mpi_size; ++i) {
            printf("%s%s", &mpi_rank_names[i * MPI_MAX_PROCESSOR_NAME], ((i == mpi_size - 1) ? "\n" : ", "));
        }
        printf("Repeat times: %d\n", repeat);
        printf("Test sizes: ");
        for (size_t i = 0; i < test_sizes.size(); i++) {
            printf("%d%s", test_sizes[i], ((i == test_sizes.size() - 1) ? "\n" : ", "));
        }
        printf("Test methods: ");
        for (size_t i = 0; i < methods.size(); i++) {
            printf("%s%s", methods[i].c_str(), ((i == methods.size() - 1) ? "\n" : ", "));
        }
        printf("Selected roots: ");
        for (size_t i = 0; i < roots.size(); i++) {
            printf("%d%s", roots[i], ((i == roots.size() - 1) ? "\n" : ", "));
        }
    }


    size_t max_size = test_sizes[test_sizes.size() - 1];
    auto test_latency = new double[test_sizes.size()](), test_throughput = new double[test_sizes.size()]();
    MPI_Request *requests = new MPI_Request[batch];
    auto buf = new uint8_t[max_size * batch * mpi_size], alt_buf = new uint8_t[max_size * batch * mpi_size];
    auto counts = new int[mpi_size];

    // test functions
    const auto do_bcast = [&](int count, int root) {
        for (int j = 0; j < batch; ++j) {
            MPI_Ibcast(buf + j * max_size, count, MPI_INT32_T, root, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    const auto do_reduce = [&](int count, int root) {
        for (int j = 0; j < batch; ++j) {
            MPI_Ireduce(buf + j * max_size, alt_buf + j * max_size, count, MPI_INT32_T, MPI_SUM, root, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    const auto do_gather = [&](int count, int root) {
        for (int j = 0; j < batch; ++j) {
            MPI_Igather(buf + j * max_size, count, MPI_INT32_T, alt_buf + j * max_size * mpi_size, count, MPI_INT32_T, root, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    const auto do_scatter = [&](int count, int root) {
        for (int j = 0; j < batch; ++j) {
            MPI_Iscatter(alt_buf + j * max_size * mpi_size, count, MPI_INT32_T, buf + j * max_size, count, MPI_INT32_T, root, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    const auto do_scan = [&](int count, [[maybe_unused]] int root) {
        for (int j = 0; j < batch; ++j) {
            MPI_Iscan(buf + j * max_size, alt_buf + j * max_size, count, MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    const auto do_reduce_scatter = [&](int count, [[maybe_unused]] int root) {
        // fill counts array
        if (counts[0] != count / mpi_size) {
            std::fill(counts, counts + mpi_size, count / mpi_size);
        }
        for (int j = 0; j < batch; ++j) {
            MPI_Ireduce_scatter(alt_buf + j * max_size * mpi_size, buf + j * max_size, counts, MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    const auto do_allreduce = [&](int count, [[maybe_unused]] int root) {
        for (int j = 0; j < batch; ++j) {
            MPI_Iallreduce(buf + j * max_size, alt_buf + j * max_size, count, MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    const auto do_allgather = [&](int count, [[maybe_unused]] int root) {
        for (int j = 0; j < batch; ++j) {
            MPI_Iallgather(buf + j * max_size, count, MPI_INT32_T, alt_buf + j * max_size * mpi_size, count, MPI_INT32_T, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    const auto do_alltoall = [&](int count, [[maybe_unused]] int root) {
        for (int j = 0; j < batch; ++j) {
            MPI_Ialltoall(buf + j * max_size * mpi_size, count, MPI_INT32_T, alt_buf + j * max_size * mpi_size, count, MPI_INT32_T, MPI_COMM_WORLD, &requests[j]);
        }
        MPI_Waitall(batch, requests, MPI_STATUSES_IGNORE);
    };

    std::unordered_map<std::string, std::function<void(int, int)>> method_map = {
        {"bcast", do_bcast},
        {"reduce", do_reduce},
        {"gather", do_gather},
        {"scatter", do_scatter},
        {"scan", do_scan},
        {"reduce_scatter", do_reduce_scatter},
        {"allreduce", do_allreduce},
        {"allgather", do_allgather},
        {"alltoall", do_alltoall},
    };

    timer timer;

    // running for different sizes
    for (const auto &m : methods) {
        DO_RANK_ZERO(puts("=================="));
        DO_RANK_ZERO(printf("Method: %s\n", m.c_str()));
        // determine method
        bool single_rooted = is_single_rooted(m);
        auto func = method_map[m];
        // determine all roots
        std::vector<int> current_roots = {0};
        if (single_rooted) {
            current_roots = roots;
        }
        // iterate over all roots
        for (auto r : current_roots) {
            DO_RANK_ZERO(printf("Root: %d%s\n", r, single_rooted ? "" : " (not used)"));
            DO_RANK_ZERO(puts("   Size    Latency(us) Bandwidth(Mbps)"));
            // fill with invalid values
            std::fill(test_latency, test_latency + test_sizes.size(), std::numeric_limits<double>::max());
            std::fill(test_throughput, test_throughput + test_sizes.size(), 0.);
            // iterate over sizes
            for (size_t s = 0; s < test_sizes.size(); ++s) {
                int size = test_sizes[s];
                int count = size / 4;
                // warmup run & official run
                for (int j = 0; j < warmup + 1; ++j) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    auto duration = timer([&](){
                        for (int i = 0; i < repeat; ++i) {
                            func(count, r);
                        }
                    });
                    MPI_Barrier(MPI_COMM_WORLD);
                    test_latency[s] = std::min((double) duration * 1e6, test_latency[s]);
                    test_throughput[s] = std::max((double) size * repeat * batch / 1000000 / duration, test_throughput[s]);
                }
                // print results
                if (mpi_rank == 0) {
                    printf("%7d %11.3f %12.5f\n", test_sizes[s], test_latency[s], test_throughput[s]);
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
