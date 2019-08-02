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
