# v0.0.4
20201008: Issues fixed by adding the proper `SystemRequirements`

Notes by ligges@statistik.tu-dortmund.de

On Mon, Oct 5, 2020 at 11:39 PM <ligges@statistik.tu-dortmund.de> wrote:
Dear maintainer,

package rTorch_0.0.4.tar.gz does not pass the incoming checks automatically, please see the following pre-tests:
Windows: <https://win-builder.r-project.org/incoming_pretest/rTorch_0.0.4_20201006_011755/Windows/00check.log>
Status: 2 ERRORs
Debian: <https://win-builder.r-project.org/incoming_pretest/rTorch_0.0.4_20201006_011755/Debian/00check.log>
Status: 1 ERROR

Last released version's CRAN status: OK: 11
See: <https://CRAN.R-project.org/web/checks/check_results_rTorch.html>

CRAN Web: <https://cran.r-project.org/package=rTorch>

Please fix all problems and resubmit a fixed version via the webform.
If you are not sure how to fix the problems shown, please ask for help on the R-package-devel mailing list:
<https://stat.ethz.ch/mailman/listinfo/r-package-devel>
If you are fairly certain the rejection is a false positive, please reply-all to this message and explain.

More details are given in the directory:
<https://win-builder.r-project.org/incoming_pretest/rTorch_0.0.4_20201006_011755/>
The files will be removed after roughly 7 days.

No strong reverse dependencies to be checked.

Best regards,
CRAN teams' auto-check service
Flavor: r-devel-linux-x86_64-debian-gcc, r-devel-windows-ix86+x86_64
Check: CRAN incoming feasibility, Result: Note_to_CRAN_maintainers
  Maintainer: 'Alfonso R. Reyes <alfonso.reyes@oilgainsanalytics.com>'

# v0.0.2
Notes submitted by Martina Schmirl

## single quotes. `DONE!`
Please always write package names, software names and API names in
single quotes in the title and the description field.
f.i.: --> 'Python'

## change dontrun to donttest. `DONE!`
\dontrun{} should be only used if the example really cannot be executed
(e.g. because of missing additional software, missing API keys, ...) by
the user. That's why wrapping examples in \dontrun{} adds the comment
("# Not run:") as a warning for the user.
Please unwrap the examples if they are executable in < 5 sec, or replace
\dontrun{} with \donttest.


## warning messages. `DONE!`
Please check:
Warning messages:
1: In res[i] <- withCallingHandlers(if (tangle) process_tangle(group)
else process_group(group),  :
   number of items to replace is not a multiple of replacement length
2: In res[i] <- withCallingHandlers(if (tangle) process_tangle(group)
else process_group(group),  :
   number of items to replace is not a multiple of replacement length
3: In res[i] <- withCallingHandlers(if (tangle) process_tangle(group)
else process_group(group),  :
   number of items to replace is not a multiple of replacement length
4: In res[i] <- withCallingHandlers(if (tangle) process_tangle(group)
else process_group(group),  :
   number of items to replace is not a multiple of replacement length


## do not use print/cat to write information to the console. `DONE!`
You write information messages to the console that cannot be easily
suppressed. Instead of `print()/cat()` rather use `message()/warning()`  or
`if(verbose)cat(..)` if you really have to write text to the console.


## add \value to .Rd files. `DONE!`
Please add \value to .Rd files and explain the functions results in the
documentation.
