#include <Rcpp.h>

extern "C"
{
    void cluster(
        const double *dists,
        int *clusters,
        int *outliers,
        const int items,
        const int num_clusters,
        const int min_size);
}

//' Robust Single-Linkage Clustering
//'
//' @param dists distance matrix (square and symmetric)
//' @param num_clusters the number of clusters to find
//' @param min_size the minimum size of a cluster
//'
//' @return
//'   \item{clusters}{vector with the cluster id (number) for each item}
//'   \item{outliers}{binary vector indicating outliers}
//' @export
//'
//' @example
//' library(rslc)
//' mat <- rbind(cbind(rnorm(10, 5), rnorm(10, 5)), cbind(rnorm(10, -5), rnorm(10, 5)), matrix(rnorm(20), 10, 2), c(-5, 0))
//' cl <- rslc(as.matrix(dist(mat)), 3, 3)
//' plot(mat[, 1], mat[, 2], col=factor(cl$clusters, 0:4), pch=cl$outliers*3+1)
//' legend("topleft", legend = c(paste("Cluster", 1:3), "Outliers"), col = factor(c(1,2,3,0)), pch=c(1,1,1,4))
//'
// [[Rcpp::export()]]
Rcpp::List rslc(const Rcpp::NumericMatrix dists, int num_clusters, int min_size)
{
    Rcpp::IntegerVector clusters(dists.ncol());
    Rcpp::LogicalVector outliers(dists.ncol());
    cluster(dists.begin(), clusters.begin(), outliers.begin(), dists.ncol(), num_clusters, min_size);
    Rcpp::List ret;
    ret["clusters"] = clusters;
    ret["outliers"] = outliers;
    return (ret);
}
