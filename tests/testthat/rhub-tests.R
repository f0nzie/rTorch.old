# remote tests using rhub
library(rhub)

to_test_on <- c("windows-x86_64-release",
                "ubuntu-gcc-release",
                "macos-highsierra-release")

check_cran <- rhub::check_for_cran(platforms = to_test_on)
check_cran$cran_summary()


previous_checks <- rhub::list_package_checks()
group_id <- previous_checks$group[1]
group_check <- rhub::get_check(group_id)
group_check
