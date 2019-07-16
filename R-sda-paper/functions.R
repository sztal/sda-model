library(tidyverse)
library(boot)


# Simple bootstrap for univariate statistics
simple_boot <- function(x, func, R = 100, ...) {
    stat <- function(x, idx) func(x[idx], ...)
    res <- boot(x, stat, R = 1000, stype = "i")
    mean(res$t[, 1])
}

# Plotting functions
add_base_theme <- function(plt) {
    plt +
        theme(
            axis.text = element_text(size = 7),
            axis.title = element_text(face = "bold"),
            axis.title.x = element_text(margin = margin(t = 15)),
            axis.title.y = element_text(margin = margin(r = 15)),
            panel.background = element_rect(fill = "gray98"),
            strip.background = element_rect(fill = "black"),
            strip.text = element_text(size = 9, color = "gray98", face = "bold")
        )
}
# Make line plot
make_line_plot <- function(df, legend_position, hline = Inf) {
    plt <- df %>%
        ggplot(aes(x = N, y = stat, color = alpha_f, group = alpha_f)) +
        geom_hline(yintercept = hline, linetype = 2, size = .5) +
        geom_ribbon(aes(ymin = lower, ymax = upper, fill = alpha_f), alpha = .1, linetype = 1, size = .2) +
        geom_line() +
        geom_point(size = 2, shape = 21, fill = "white") +
        scale_x_log10(breaks = SIZES) +
        annotation_logticks(sides = "b") +
        scale_color_manual(values = ALPHA_COLORS, labels = ALPHA_LABELS) +
        scale_fill_manual(values = ALPHA_COLORS, labels = ALPHA_LABELS)  +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5),
            legend.position = legend_position,
            legend.background = element_rect(fill = "white", color = "black"),
            legend.text = element_text(size = 9, face = "bold"),
            legend.title.align = .5,
            legend.direction = "horizontal"
        ) +
        guides(fill = FALSE) +
        labs(
            x = "Network size (N)",
            color = "α"
        )
    plt %>%
        add_base_theme
}
