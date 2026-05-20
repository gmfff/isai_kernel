#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

void analyze_pair_lengths(
    int nb,
    const std::vector<int>& h_Mbrow,
    const std::vector<int>& h_PairPtr)
{
    auto percentile_int = [](std::vector<int> v, double p) -> int {
        if (v.empty()) return 0;

        std::sort(v.begin(), v.end());

        size_t id = static_cast<size_t>((v.size() - 1) * p);
        return v[id];
    };

    int Mnnz = h_Mbrow[nb];

    std::vector<int> colN;
    std::vector<int> colQ;
    std::vector<int> allQlen;

    colN.reserve(nb);
    colQ.reserve(nb);
    allQlen.reserve(Mnnz);

    long long totalN = 0;
    long long totalQ = 0;

    int maxN = 0;
    int maxQ = 0;
    int maxQlen = 0;

    for (int j = 0; j < nb; ++j) {

        int start = h_Mbrow[j];
        int end   = h_Mbrow[j + 1];

        int N = end - start;

        colN.push_back(N);

        totalN += N;
        maxN = std::max(maxN, N);

        int Qj = 0;

        for (int g0 = start; g0 < end; ++g0) {

            int qlen = h_PairPtr[g0 + 1] - h_PairPtr[g0];

            allQlen.push_back(qlen);

            Qj += qlen;
            totalQ += qlen;

            maxQlen = std::max(maxQlen, qlen);
        }

        colQ.push_back(Qj);

        maxQ = std::max(maxQ, Qj);
    }

    std::cout << "\n========================================\n";
    std::cout << "        Pair Length Statistics\n";
    std::cout << "========================================\n";

    std::cout << "nb            = " << nb << "\n";
    std::cout << "Mnnz          = " << Mnnz << "\n";
    std::cout << "total_pairs   = " << totalQ << "\n";

    std::cout << "\n---- N per column ----\n";

    std::cout << "avg N   = " << (double)totalN / nb << "\n";
    std::cout << "max N   = " << maxN << "\n";
    std::cout << "p50 N   = " << percentile_int(colN, 0.50) << "\n";
    std::cout << "p90 N   = " << percentile_int(colN, 0.90) << "\n";
    std::cout << "p95 N   = " << percentile_int(colN, 0.95) << "\n";
    std::cout << "p99 N   = " << percentile_int(colN, 0.99) << "\n";

    std::cout << "\n---- Q per column (sum qlen) ----\n";

    std::cout << "avg Q   = " << (double)totalQ / nb << "\n";
    std::cout << "max Q   = " << maxQ << "\n";
    std::cout << "p50 Q   = " << percentile_int(colQ, 0.50) << "\n";
    std::cout << "p90 Q   = " << percentile_int(colQ, 0.90) << "\n";
    std::cout << "p95 Q   = " << percentile_int(colQ, 0.95) << "\n";
    std::cout << "p99 Q   = " << percentile_int(colQ, 0.99) << "\n";

    std::cout << "\n---- qlen per idx ----\n";

    double avgQlen =
        allQlen.empty() ? 0.0 :
        (double)totalQ / allQlen.size();

    std::cout << "avg qlen = " << avgQlen << "\n";
    std::cout << "max qlen = " << maxQlen << "\n";
    std::cout << "p50 qlen = " << percentile_int(allQlen, 0.50) << "\n";
    std::cout << "p90 qlen = " << percentile_int(allQlen, 0.90) << "\n";
    std::cout << "p95 qlen = " << percentile_int(allQlen, 0.95) << "\n";
    std::cout << "p99 qlen = " << percentile_int(allQlen, 0.99) << "\n";

    long long hist[10] = {0};

    for (int x : allQlen) {

        if (x == 0) hist[0]++;
        else if (x == 1) hist[1]++;
        else if (x < 4) hist[2]++;
        else if (x < 8) hist[3]++;
        else if (x < 16) hist[4]++;
        else if (x < 32) hist[5]++;
        else if (x < 64) hist[6]++;
        else if (x < 128) hist[7]++;
        else if (x < 256) hist[8]++;
        else hist[9]++;
    }

    const char* names[10] = {
        "0",
        "1",
        "2-3",
        "4-7",
        "8-15",
        "16-31",
        "32-63",
        "64-127",
        "128-255",
        ">=256"
    };

    std::cout << "\n---- qlen histogram ----\n";

    for (int i = 0; i < 10; ++i) {

        double ratio =
            allQlen.empty() ? 0.0 :
            100.0 * hist[i] / allQlen.size();

        std::cout
            << names[i]
            << " : "
            << hist[i]
            << "  ("
            << ratio
            << "%)\n";
    }

    std::cout << "========================================\n\n";
}
struct ColQInfo {
    int col;
    int N;
    int Q;
    int max_qlen;
    double avg_qlen;
};

void analyze_column_q_distribution(
    int nb,
    const std::vector<int>& h_Mbrow,
    const std::vector<int>& h_PairPtr,
    int topK = 20)
{
    std::vector<ColQInfo> infos;
    infos.reserve(nb);

    for (int j = 0; j < nb; ++j) {
        int start = h_Mbrow[j];
        int end   = h_Mbrow[j + 1];
        int N     = end - start;

        int Qj = 0;
        int max_qlen = 0;

        for (int g0 = start; g0 < end; ++g0) {
            int qlen = h_PairPtr[g0 + 1] - h_PairPtr[g0];
            Qj += qlen;
            max_qlen = std::max(max_qlen, qlen);
        }

        double avg_qlen = (N > 0) ? (double)Qj / N : 0.0;

        infos.push_back({j, N, Qj, max_qlen, avg_qlen});
    }

    std::sort(
        infos.begin(),
        infos.end(),
        [](const ColQInfo& a, const ColQInfo& b) {
            return a.Q > b.Q;
        }
    );

    if (topK > (int)infos.size()) topK = (int)infos.size();

    std::cout << "\n==== Top columns by total qlen Qj ====\n";
    std::cout << "rank\tcol\tN\tQj\tavg_qlen\tmax_qlen\n";

    for (int i = 0; i < topK; ++i) {
        const auto& x = infos[i];

        std::cout
            << i
            << "\t"
            << x.col
            << "\t"
            << x.N
            << "\t"
            << x.Q
            << "\t"
            << x.avg_qlen
            << "\t\t"
            << x.max_qlen
            << "\n";
    }

    std::cout << "======================================\n\n";
}

void analyze_column_q_bins(
    int nb,
    const std::vector<int>& h_Mbrow,
    const std::vector<int>& h_PairPtr,
    int numBins = 10)
{
    std::vector<long long> binQ(numBins, 0);
    std::vector<int> binCols(numBins, 0);
    std::vector<int> binMaxQ(numBins, 0);

    long long totalQ = 0;

    for (int j = 0; j < nb; ++j) {
        int start = h_Mbrow[j];
        int end   = h_Mbrow[j + 1];

        int Qj = 0;

        for (int g0 = start; g0 < end; ++g0) {
            Qj += h_PairPtr[g0 + 1] - h_PairPtr[g0];
        }

        int b = (long long)j * numBins / nb;
        if (b >= numBins) b = numBins - 1;

        binQ[b] += Qj;
        binCols[b]++;
        binMaxQ[b] = std::max(binMaxQ[b], Qj);

        totalQ += Qj;
    }

    std::cout << "\n==== Column-index distribution of Qj ====\n";
    std::cout << "bin\tcol_range\tcols\tsumQ\tavgQ\tmaxQ\tpercent_totalQ\n";

    for (int b = 0; b < numBins; ++b) {
        int l = (long long)b * nb / numBins;
        int r = (long long)(b + 1) * nb / numBins - 1;

        double avgQ = binCols[b] ? (double)binQ[b] / binCols[b] : 0.0;
        double pct  = totalQ ? 100.0 * binQ[b] / totalQ : 0.0;

        std::cout
            << b << "\t"
            << "[" << l << "," << r << "]\t"
            << binCols[b] << "\t"
            << binQ[b] << "\t"
            << avgQ << "\t"
            << binMaxQ[b] << "\t"
            << pct << "%\n";
    }

    std::cout << "========================================\n\n";
}
void analyze_last_columns(
    int nb,
    const std::vector<int>& h_Mbrow,
    const std::vector<int>& h_PairPtr,
    int lastK = 30)
{
    int begin = std::max(0, nb - lastK);

    std::cout << "\n==== Last Columns Statistics ====\n";
    std::cout << "col\tN\tQj\tavg_qlen\tmax_qlen\n";

    for (int j = begin; j < nb; ++j) {

        int start = h_Mbrow[j];
        int end   = h_Mbrow[j + 1];

        int N = end - start;

        int Qj = 0;
        int max_qlen = 0;

        for (int g0 = start; g0 < end; ++g0) {

            int qlen =
                h_PairPtr[g0 + 1] - h_PairPtr[g0];

            Qj += qlen;

            max_qlen = std::max(max_qlen, qlen);
        }

        double avg_qlen =
            (N > 0) ? (double)Qj / N : 0.0;

        std::cout
            << j
            << "\t"
            << N
            << "\t"
            << Qj
            << "\t"
            << avg_qlen
            << "\t\t"
            << max_qlen
            << "\n";
    }

    std::cout << "=================================\n\n";
}