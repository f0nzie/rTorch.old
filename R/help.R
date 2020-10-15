

register_torch_help_handler <- function() {
    # this needs to be connected to the correct API
    # right now is a dummy
    reticulate::register_module_help_handler("torch", function(name, subtopic = NULL) {

        # get the base pytorch help url
        version <- torch$`__version__`
        version <- strsplit(version, ".", fixed = TRUE)[[1]]
        # help_url <- paste0("https://www.pytorch.org/versions/r",
                           # version[1], ".", version[2], "/api_docs/python/")
        help_url <- paste0("https://www.pytorch.org/docs/",
                           version[1], ".", version[2], ".", version[3])

        # some adjustments
        name <- sub("^torch", "torch", name)
        name <- sub("python.client.session.", "", name, fixed = TRUE)
        name <- sub("python.ops.", "", name, fixed = TRUE)
        if (grepl("torch.contrib.opt", name)) {
            components <- strsplit(name, ".", fixed = TRUE)[[1]]
            class_name <- components[[length(components)]]
            name <- paste0("torch.contrib.opt", ".", class_name)
        }

        # form topic url
        topic_url <- gsub(".", "/", name, fixed = TRUE)
        if (!is.null(subtopic))
            topic_url <- paste0(topic_url, "#", subtopic)

        # return the full url
        paste0(help_url, topic_url)
    })
}

