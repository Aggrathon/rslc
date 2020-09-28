context("RSLC")

test_that("RSLC finds obvious clusters and outliers", {
    mat <- rbind(
        cbind(rnorm(10, 10), rnorm(10, 10)),
        cbind(rnorm(10, -10), rnorm(10, 10)),
        matrix(rnorm(20), 10, 2),
        c(-20, -20)
    )
    cl <- rslc(as.matrix(dist(mat)), 3, 3)
    expect_equal(sd(cl$clusters[1:10]), 0)
    expect_equal(sd(cl$clusters[11:20]), 0)
    expect_equal(sd(cl$clusters[21:30]), 0)
    expect_false(any(cl$outliers[1:30]))
    expect_true(cl$outliers[31])
})
