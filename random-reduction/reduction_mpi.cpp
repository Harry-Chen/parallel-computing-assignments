#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define epsilon 1.e-8

using namespace std;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int M, N;

    string T, P, Db;
    M = atoi(argv[1]);
    N = atoi(argv[2]);

    double elapsedTime, elapsedTime2;
    timeval start, end, end2;

    if (argc > 3) {
        T = argv[3];
        if (argc > 4) {
            P = argv[4];
            if (argc > 5) {
                Db = argv[5];
            }
        }
    }
    // cout<<T<<P<<endl;

    double **U_t;
    double **Alphas, **Betas, **Gammas;

    int acum = 0;
    int temp1, temp2;

    U_t = new double *[M];
    Alphas = new double *[M];
    Betas = new double *[M];
    Gammas = new double *[M];

    for (int i = 0; i < M; i++) {
        U_t[i] = new double[N];
        Alphas[i] = new double[M];
        Betas[i] = new double[M];
        Gammas[i] = new double[M];
    }

    // Read from file matrix, if not available, app quit
    // Already transposed

    ifstream matrixfile("matrix");
    if (!(matrixfile.is_open())) {
        cout << "Error: file not found" << endl;
        return 0;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrixfile >> U_t[i][j];
        }
    }

    matrixfile.close();

    /* Reductions */

    gettimeofday(&start, NULL);

    // slice the array along M
    int M_slice = (M - 1) / mpi_size + 1;
    int M_start = mpi_rank * M_slice;
    int M_end = std::min(M, (mpi_rank + 1) * M_slice);
    int M_size_round = mpi_size * M_slice;

    MPI_Request requests[3];
    auto *alpha_temp = new double[mpi_rank == 0 ? M_size_round : M_slice], *beta_temp = new double[mpi_rank == 0 ? M_size_round : M_slice];
    auto *gamma_temp = new double[mpi_rank == 0 ? M_size_round * M : M_slice * M];

    // calculate gamma (blocking along i) & gather
    for (int i = M_start; i < M_end; i++) {
        int i_ = i % M_slice;
        for (int j = 0; j < M; j++) {
            double gamma = 0.0;
            for (int k = 0; k < N; k++) {
                gamma += U_t[i][k] * U_t[j][k];
            }
            gamma_temp[i_ * M + j] = gamma;
        }
    }
    MPI_Igather(mpi_rank == 0 ? MPI_IN_PLACE : gamma_temp, M * M_slice, MPI_DOUBLE, gamma_temp, M * M_slice, MPI_DOUBLE, 0, MPI_COMM_WORLD, &requests[2]);


    // calculate alpha & gather
    for (int i = M_start; i < M_end; ++i) {
        int i_ = i % M_slice;
        double alpha = 0.0;
        for (int k = 0; k < N; ++k) {
             alpha += U_t[i][k] * U_t[i][k];
        }
        alpha_temp[i_] = alpha;
    }
    MPI_Igather(mpi_rank == 0 ? MPI_IN_PLACE : alpha_temp, M_slice, MPI_DOUBLE, alpha_temp, M_slice, MPI_DOUBLE, 0, MPI_COMM_WORLD, &requests[0]);

    // calculate beta & gather
    for (int j = M_start; j < M_end; ++j) {
        int j_ = j % M_slice;
        double beta = 0.0;
        for (int k = 0; k < N; ++k) {
             beta += U_t[j][k] * U_t[j][k];
        }
        beta_temp[j_] = beta;
    }
    MPI_Igather(mpi_rank == 0 ? MPI_IN_PLACE : beta_temp, M_slice, MPI_DOUBLE, beta_temp, M_slice, MPI_DOUBLE, 0, MPI_COMM_WORLD, &requests[1]);

    // wait for all communication to finish
    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);

    gettimeofday(&end, NULL);

    // fix final result (copy back from buffer)
    if (mpi_rank == 0) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                Gammas[i][j] = gamma_temp[i * M + j];
                Betas[i][j] = beta_temp[j];
                Alphas[i][j] = alpha_temp[i];
            }
        }
    }

    // output only on first rank
    if (mpi_rank == 0) {

        // Output time and iterations
        if (T == "-t" || P == "-t") {
            elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
            elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
            cout << "Time: " << elapsedTime << " ms." << endl << endl;
        }

        // Output the matrixes for debug
        if (T == "-p" || P == "-p") {
            cout << "Alphas" << endl << endl;
            for (int i = 0; i < M; i++) {

                for (int j = 0; j < M; j++) {

                    cout << Alphas[i][j] << "  ";
                }
                cout << endl;
            }

            cout << endl << "Betas" << endl << endl;
            for (int i = 0; i < M; i++) {

                for (int j = 0; j < M; j++) {
                    cout << Betas[i][j] << "  ";
                }
                cout << endl;
            }

            cout << endl << "Gammas" << endl << endl;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {

                    cout << Gammas[i][j] << "  ";
                }
                cout << endl;
            }
        }

        // Generate files for debug purpouse
        if (Db == "-d" || T == "-d" || P == "-d") {

            ofstream Af;
            // file for Matrix A
            Af.open("AlphasMPI.mat");
            /*    Af<<"# Created from debug\n# name: A\n# type: matrix\n# rows:
            * "<<M<<"\n# columns: "<<N<<"\n";*/

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    Af << " " << Alphas[i][j];
                }
                Af << "\n";
            }

            Af.close();

            ofstream Uf;

            // File for Matrix U
            Uf.open("BetasMPI.mat");
            /*    Uf<<"# Created from debug\n# name: Ugpu\n# type: matrix\n# rows:
            * "<<M<<"\n# columns: "<<N<<"\n";*/

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    Uf << " " << Betas[i][j];
                }
                Uf << "\n";
            }
            Uf.close();

            ofstream Vf;
            // File for Matrix V
            Vf.open("GammasMPI.mat");
            /*    Vf<<"# Created from debug\n# name: Vgpu\n# type: matrix\n# rows:
            * "<<M<<"\n# columns: "<<N<<"\n";*/

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    Vf << " " << Gammas[i][j];
                }
                Vf << "\n";
            }

            Vf.close();

            ofstream Sf;
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);

    // free pointers
    for (int i = 0; i < M; i++) {
        delete[] Alphas[i];
        delete[] U_t[i];
        delete[] Betas[i];
        delete[] Gammas[i];
    }
    delete[] Alphas;
    delete[] Betas;
    delete[] Gammas;
    delete[] U_t;
    delete[] alpha_temp;
    delete[] beta_temp;
    delete[] gamma_temp;

    MPI_Finalize();

    return 0;
}
