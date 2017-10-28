# dofs metrics

library(tidyverse)
library(gridExtra)
library(grid)

dofs_metrics <- read_csv("dofs-metrics_20171028.csv")
dofs_metrics$ninfom <- factor(dofs_metrics$ninfom)
dofs_metrics <- dofs_metrics %>%
  mutate(compact_ratio = as.numeric(as.character(compact))/as.numeric(as.character(ninfom)))

grid_arrange_shared_legend <- function(..., ncol = length(list(...)), nrow = 1, position = c("bottom", "right")) {
  #' https://github.com/tidyverse/ggplot2/wiki/Share-a-legend-between-two-ggplot2-graphs
  plots <- list(...)
  position <- match.arg(position)
  g <- ggplotGrob(plots[[1]] + theme(legend.position = position))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  lwidth <- sum(legend$width)
  gl <- lapply(plots, function(x) x + theme(legend.position="none"))
  gl <- c(gl, ncol = ncol, nrow = nrow)
  
  combined <- switch(position,
                     "bottom" = arrangeGrob(do.call(arrangeGrob, gl),
                                            legend,
                                            ncol = 1,
                                            heights = unit.c(unit(1, "npc") - lheight, lheight)),
                     "right" = arrangeGrob(do.call(arrangeGrob, gl),
                                           legend,
                                           ncol = 2,
                                           widths = unit.c(unit(1, "npc") - lwidth, lwidth)))
  
  grid.newpage()
  grid.draw(combined)
  
  # return gtable invisibly
  invisible(combined)
  
}

p1 <- ggplot(dofs_metrics, 
       aes(x=ninfom, 
           y=inform_perc, 
           group=Algorithm, 
           color=Algorithm))+
  geom_smooth(se=FALSE) +
  xlab("# Informative Features") +
  ylab("Perc. Inform. Feats") + 
  theme_bw()

p2 <- ggplot(dofs_metrics, 
       aes(x=ninfom, 
           y=ratio, 
           group=Algorithm, 
           color=Algorithm))+
  geom_smooth(se=FALSE) + 
  xlab("# Informative Features") +
  ylab("Ratio Inform. Feats") +
  theme_bw()

p3 <- ggplot(dofs_metrics, 
             aes(x=ninfom, 
                 y=compact, 
                 group=Algorithm, 
                 color=Algorithm))+
  geom_smooth(se=FALSE) +
  xlab("# Informative Features") + 
  ylab("Compactness") +
  theme_bw()


p4 <- ggplot(dofs_metrics, 
             aes(x=ninfom, 
                 y=saliency, 
                 group=Algorithm, 
                 color=Algorithm))+
  geom_smooth(se=FALSE) +
  xlab("# Informative Features") + 
  ylab("Saliency") +
  theme_bw()

grid_arrange_shared_legend(p1, p2, p3, p4, ncol = 2, nrow = 2)
grid_arrange_shared_legend(p1, p2, ncol = 2, nrow = 1)
