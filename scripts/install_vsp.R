repos <- c(CRAN = "https://cloud.r-project.org")

pkgs <- c("vsp")

to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]

if (length(to_install) > 0) {
  install.packages(to_install, repos = repos, dependencies = TRUE)
}

cat("R packages ready:\n")
cat("vsp version:", as.character(utils::packageVersion("vsp")), "\n")