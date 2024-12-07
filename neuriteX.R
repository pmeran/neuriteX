
# © 2024 Paul Meraner
# neuriteX.py


# 4. Visualization of neurite integrity scores (R)

# 4.1 N_perc

library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
library(dplyr)

# INSERT absolute path for R project folder:
setwd(<R project folder>)



df <- read.table("df_R1.csv", header = TRUE, sep = ",")

W <- as.vector(unique(df$well))

pL <- list()

for (well in W) {
  ix <- which(df$well == well)
  df_x = df[ix,]
  T = paste0(df_x$well[1], ' ', df_x$condition[1])
  
  pL[[well]] <- ggplot(df_x, aes(x=time_n, y = N_perc, colour=state)) +
  geom_point(size=2, show.legend=FALSE) +
  # ggtitle(T) + # for the main title
  theme(plot.title = element_text(size=14)) +
  xlab('time (hours)') + # for the x axis label
  ylab('N_perc')+ # for the y axis label
  theme(axis.text=element_text(size=14), axis.title=element_text(size=14), plot.title=element_text(size=14)) +
  scale_colour_manual(values = c('red','black')) +
  
  # scale_y_continuous(name=NULL, limits=c(0,40), breaks = c(0,10,20,30,40)) +
  # scale_x_continuous(name=NULL, limits=c(0,32), breaks = c(0,10,20,30)) +
  scale_y_continuous(limits=c(0,40), breaks = c(0,10,20,30,40)) +
  scale_x_continuous(limits=c(0,32), breaks = c(0,10,20,30)) +
  theme(panel.background = element_rect(fill = 'white', colour = 'white')) +
  theme(axis.line.x = element_line(color="black", linewidth = 0.4),
      axis.line.y = element_line(color="black", linewidth = 0.4)) +
  theme(text=element_text(size=14)) +
  theme(aspect.ratio=1.5)
}

grid.arrange(grobs=c(pL['B1']), ncol=1)
ggsave('N_perc.pdf', plot=last_plot(), device='pdf', width=6, height=10, units='cm', dpi=300)

# end of: # 4.1 N_perc
##############################################################


##############################################################

# 4.2 N_perc_corr

library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
library(dplyr)

df <- read.table("df_R2.csv", header = TRUE, sep = ",")

W <- as.vector(unique(df$well))

pL <- list()

for (well in W) {
  ix <- which(df$well == well)
  df_x = df[ix,]
  T = paste0(df_x$well[1], ' ', df_x$condition[1])
  
  pL[[well]] <- ggplot(df_x, aes(x=time_n, y = N_perc_corr, colour=state)) +
    geom_point(size=2, show.legend=FALSE) +
    # ggtitle(T) + # for the main title
    theme(plot.title = element_text(size=14)) +
    xlab('time (hours)') + # for the x axis label
    ylab('N_perc_corr') + # for the y axis label
    theme(axis.text=element_text(size=14), axis.title=element_text(size=14),
          plot.title=element_text(size=14)) +
    scale_colour_manual(values = c('red','black')) +
    
    scale_y_continuous(limits=c(0,40), breaks = c(0,10,20,30,40)) +
    scale_x_continuous(limits=c(0,32), breaks = c(0,10,20,30)) +
    theme(panel.background = element_rect(fill = 'white', colour = 'white')) +
    theme(axis.line.x = element_line(color="black", linewidth = 0.4),
          axis.line.y = element_line(color="black", linewidth = 0.4)) +
    theme(text=element_text(size=14)) +
    theme(aspect.ratio=1.5)
}

grid.arrange(grobs=c(pL['B1']), ncol=1)
ggsave('N_perc_corr.pdf', plot=last_plot(), device='pdf', width=6, height=10, units='cm', dpi=300)

# end of: # 4.2 N_perc_corr
############################################################



######################################################################
# 4.3 N_perc_corr

library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
library(dplyr)

df <- read.table("df_R3.csv", header = TRUE, sep = ",")

wellgroup_list <- list(c('B1'))
condition_list <- list('no-insert')

pL <- list()

for (i in 1:length(wellgroup_list)) {
  df_x <- df[df$well %in% unlist(wellgroup_list[i]),]
  df_x <- df_x %>%
    group_by(time_n, state) %>%
    mutate(boxgroup = cur_group_id())
  
  df_out <- df_x %>% group_by(time_n, state)  %>%
    summarise(
      # time_n = time_n,
      #         NII = NII,
      lo = unname(quantile(NII_well, 0.1)),
      mid = unname(quantile(NII_well, 0.5)),
      hi = unname(quantile(NII_well, 0.1)), .groups = 'drop')
  df_out <- data.frame(df_out)
  xD = 0.3 # shift off center for: geom_linerange rangebar (vertical)
  xE = 1.8 # shift off center for: geom_linerange mid lines (horizontal)
  xDg = 1.2 # geom_point shift off center
  
  T = condition_list[i]

  pL[[i]] <- ggplot() +
    
    geom_segment(data=df_out[df_out$state=='uncut',],
                 aes(x=time_n-xE, y=mid, xend=time_n+xE, yend=mid),
                 linewidth=0.7, colour='#000000') +
    geom_segment(data=df_out[df_out$state=='cut',],
                 aes(x=time_n-xE+xDg, y=mid, xend=time_n+xE+xDg, yend=mid),
                 linewidth=0.7, colour='#FF0000') +
    geom_boxplot(data=df_x[df_x$state=='uncut',],
                 aes(x=time_n, y=NII_well, group=factor(boxgroup)), color='black',
                 position='identity', lwd=0.5, outlier.shape=NA, width=2) + # was 1.5
    geom_boxplot(data=df_x[df_x$state=='cut',],
                 aes(x=time_n+xDg, y=NII_well, group=factor(boxgroup)), color='red',
                 position='identity', lwd=0.5, outlier.shape=NA, width=2) + # was 1.5
    geom_point(data=df_x[df_x$state=='uncut',],
               aes(x=time_n, y=NII_well, color=factor(state)), size=3, stroke=NA,
               show.legend=FALSE, alpha=0.4) + 
    # position=position_jitter(width=0.5, height=0)) +
    geom_point(data=df_x[df_x$state=='cut',],
               aes(x=time_n+xDg, y=NII_well, color=factor(state)), size=3, stroke=NA,
               show.legend=FALSE, alpha = 0.4) +
    #position=position_jitter(width=0.5, height=0)) +
    scale_color_manual(values = c('black', 'red')) +
    
    scale_y_continuous(limits=c(0.1,1.2), breaks = c(0.1,1.0)) +
    scale_x_continuous(limits=c(0,35), breaks = c(0,10,20,30)) + # name=NULL to suppress axis title
    # ggtitle(T) + # for the main title
    # theme(plot.title = element_text(size=9)) +
    xlab('time (h)') + # for the x axis label
    ylab('NII') + # for the y axis label
    # scale_color_manual(values=c('red', 'black')) +
    theme(axis.text=element_text(size=16), axis.title=element_text(size=16), plot.title=element_text(size=16)) +
    theme(panel.background = element_rect(fill = 'white', colour = 'white')) +
    theme(axis.line.x = element_line(color="black", linewidth = 0.4),
          axis.line.y = element_line(color="black", linewidth = 0.4)) +
    theme(aspect.ratio=1.5)
}

G = c(pL[1])

grid.arrange(grobs=G, ncol=1)
ggsave('NII.pdf', plot=last_plot(), device='pdf', width=7, height=9, units='cm', dpi=300)

# end of: # 4.3 N_perc_corr
######################################################################

