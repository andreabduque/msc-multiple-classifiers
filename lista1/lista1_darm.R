bag_jc1 <- c(0.30508475,0.34527687,0.3713355,0.32467532,0.35294118,0.35625,
         0.33663366,0.31229236,0.30872483,0.33887043)
rs_jc1 <- c(0.28880866,0.33112583,0.34027778,0.32653061,0.30927835,0.30769231,
             0.34146341,0.28469751,0.29268293,0.31141869)

wilcox.test(bag_jc1, rs_jc1,  alternative = c("greater"),  conf.level = 0.99, paired=TRUE)

  library(ggpubr)
score = c(bag_jc1, rs_jc1)
method = c(rep('Bagging (100%)', times=10), rep('Random Subspace (50%)', times=10))

ggboxplot(data.frame(method, score), x = "method", y = "score", 
          color = "method", palette = c("red", "blue"),
          order = c("Bagging (100%)", "Random Subspace (50%)"),
         
          ylab = "F-score", xlab = "Técnicas de ensemble com árvores de decisão para base JM1")
  


perc7 = c(0.32786885,0.30819672,0.36,0.33112583,0.34899329,0.31229236
          ,0.35714286,0.31649832,0.29452055,0.33220339)

perc1 = c(0.32666667,0.34591195,0.34868421,0.34193548,0.35294118,0.38095238
          ,0.35215947,0.31788079,0.32679739,0.3557047)
