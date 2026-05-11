pca_varimax_from_components <- function(components, normalize = FALSE, eps = 1e-5) {
  # components: dim x p, same as sklearn PCA.components_
  components <- as.matrix(components)

  # R varimax expects loadings matrix: p x dim
  L <- t(components)

  vm <- stats::varimax(L, normalize = normalize, eps = eps)

  rotated <- unclass(vm$loadings)  # p x dim
  rotmat <- unclass(vm$rotmat)     # dim x dim

  list(
    rotated = rotated,
    rotmat = rotmat
  )
}