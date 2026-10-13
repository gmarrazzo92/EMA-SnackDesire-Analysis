# Faithful extraction of the S1 Fig plot from
# the original peri-snack plotting code. Only the S1 figure is generated.
# Run via run_all.py or:
# EMA_RELEASE_ROOT=/path/to/release Rscript scripts/11_make_S1_figure.R
root <- Sys.getenv('EMA_RELEASE_ROOT',unset='')
if (!nzchar(root)) stop('Set EMA_RELEASE_ROOT or use run_all.py')
for (pkg in c('dplyr','readr','ggplot2')) if (!requireNamespace(pkg,quietly=TRUE)) stop('Missing R package: ',pkg)
suppressPackageStartupMessages({library(dplyr);library(readr);library(ggplot2)})
traj <- readr::read_csv(file.path(root,'results','08_perisnack_tables','participant_balanced_trajectories.csv'),show_col_types=FALSE)
PRIMARY_MATCH_SET <- 'primary_time_only_no_reuse'
BASE_FAMILY <- 'sans'
col_snack <- '#BB4430'; col_control <- '#3D6A9A'; muted <- '#6B7078'; ink <- '#1A1A1A'; gridcol <- '#E6E6E6'
theme_manuscript <- function(base_size=11,grid='y') {
 th <- theme_minimal(base_size=base_size,base_family=BASE_FAMILY)+theme(
  text=element_text(colour=ink),plot.title=element_text(face='bold',size=base_size+1,margin=margin(b=4)),
  plot.title.position='plot',plot.subtitle=element_text(colour=muted,size=base_size-.5,margin=margin(b=8)),
  axis.title=element_text(size=base_size-.5),axis.title.x=element_text(margin=margin(t=6)),
  axis.title.y=element_text(margin=margin(r=6)),axis.text=element_text(colour=muted,size=base_size-1.5),
  panel.grid.minor=element_blank(),panel.grid.major=element_line(colour=gridcol,linewidth=.4),
  strip.text=element_text(face='bold',size=base_size-.5,margin=margin(b=5)),
  legend.position='top',legend.justification='left',legend.title=element_text(size=base_size-1),
  legend.text=element_text(size=base_size-1),legend.key.width=unit(26,'pt'),plot.margin=margin(10,14,8,12))
 if(grid=='y') th <- th+theme(panel.grid.major.x=element_blank())
 th
}
var_label <- function(x) dplyr::recode(as.character(x),'CD'='Snack desire intensity',
 'LowControl'='Low perceived control','Stress'='Stress',.default=as.character(x))
traj_primary <- traj %>% filter(match_set==PRIMARY_MATCH_SET,variable %in% c('CD','LowControl','Stress')) %>%
 mutate(variable_label=factor(var_label(variable), levels=c('Snack desire intensity','Low perceived control','Stress')),
 condition_label=factor(condition,levels=c('Matched non-snack','Snack window')))
ord <- c('Matched non-snack','Snack window')
cond_col <- c('Matched non-snack'=col_control,'Snack window'=col_snack)
cond_lty <- c('Matched non-snack'='31','Snack window'='solid')
cond_shp <- c('Matched non-snack'=16,'Snack window'=17)
offs <- sort(unique(traj_primary$offset))
off_lab <- ifelse(offs==0,'t0',paste0('t',ifelse(offs>0,'+','\u2212'),abs(offs)))
yr <- range(c(traj_primary$ci95_low,traj_primary$ci95_high),na.rm=TRUE)
ypad <- diff(yr)*.10
fig <- ggplot(traj_primary,aes(offset,mean,colour=condition_label,group=condition_label))+
 geom_vline(xintercept=0,linetype='dotted',colour=muted,linewidth=.4)+
 geom_ribbon(aes(ymin=ci95_low,ymax=ci95_high,fill=condition_label),alpha=.16,colour=NA)+
 geom_line(aes(linetype=condition_label),linewidth=1)+
 geom_point(aes(shape=condition_label),size=2.5)+facet_wrap(~variable_label,nrow=1)+
 scale_colour_manual(values=cond_col,breaks=ord,name=NULL)+
 scale_fill_manual(values=cond_col,breaks=ord,name=NULL)+
 scale_linetype_manual(values=cond_lty,breaks=ord,name=NULL)+
 scale_shape_manual(values=cond_shp,breaks=ord,name=NULL)+
 scale_x_continuous(breaks=offs,labels=off_lab,expand=expansion(mult=.06))+
 scale_y_continuous(limits=c(yr[1]-ypad,yr[2]+ypad))+
 labs(title='Figure S1. Peri-snack patterns of reported desire intensity, low perceived control, and stress',
 subtitle='Participant-balanced means; bands are 95% CIs across participants',
 x='Prompt relative to focal window',y='Mean EMA rating')+
 theme_manuscript(grid='y')+theme(plot.title=element_text(size=13),panel.spacing.x=unit(14,'pt'))
out <- file.path(root,'results','11_make_S1_figure')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
ggsave(file.path(out,'S1_Fig_perisnack_trajectories_R.png'),fig,width=10.5,height=4.7,dpi=350,bg='white')
ggsave(file.path(out,'S1_Fig_perisnack_trajectories_R.pdf'),fig,width=10.5,height=4.7,device=cairo_pdf,bg='white')
message('Saved faithful R reconstruction of original S1 Fig: ',out)
