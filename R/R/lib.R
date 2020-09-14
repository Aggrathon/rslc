
#' TODO: Documentation
rslc <- function(distance_matrix, num_clusters, min_size = num_clusters**-2) {
    stopifnot(length(dim(distance_matrix)))
    stopifnot(all.equal(dim(distance_matrix)))
    if (min_size < 1.0 && min_size > 0.0)
        min_size <- as.integer(nrow(distance_matrix) * min_size)
    else
        min_size <- as.integer(min_size)
    num_clusters <- as.integer(num_clusters)
    if (is.integer(distance_matrix))
        .Call("__wrap__rslc_i32", distance_matrix, num_clusters, min_size)
    else if (is.double(distance_matrix))
        .Call("__wrap__rslc_f64", distance_matrix, num_clusters,min_size)
    else
        stop("Unknown type of distance matrix (should be integer or double)")
}
