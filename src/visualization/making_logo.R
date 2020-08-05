### BC Stats Capstone project - logo

# load libraries
library(wordcloud)
library(ggraph)
library(hexSticker)

set.seed(1) # for reproducibility

# assigning weigths for the size and color or the words
words <- data.frame(Word=c("WES", "BC Stats", "questions", "open ended", "diversity", "executives", "benefits", "decisions", "communication", "support", "leadership", "development", "value", "training", "flexible", "service", "people", "home", "salary", "job", "reports", "better", "employees", "level", "office", "hiring", "organization", "staff", "improved", "pay", "mission", "change", "managament", "orientation", "abilities", "team", "public", "space", "health", "computer", "supervisor", "tools", "process", "skills", "opportunities", "workload", "enpowerment", "career", "collaborate", "stress"), 
                              Repetitions=c(21, 14, 10, 10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))

# generate word cloud
ggsave("../../reports/figures/logo_wordcloud.png", plot=wordcloud(words = words$Word, freq = words$Repetitions, min.freq = 1,
                                       max.words=50, random.order=FALSE, rot.per=.23,
                                       colors=brewer.pal(8, "Blues")))

# making the logo with R stickers
bc_stats_sticker <- sticker("../../reports/figures/logo_wordcloud.png",
        s_x=1, s_y=1, s_width=1.80, s_height=1.75, #s_width=0.92, s_height=0.905,
        package="",
        p_size=6, p_x=1, p_y=1,
        h_fill="white",
        h_color="midnightblue",
        h_size=1.5,
        filename="../../reports/figures/logo.png")
bc_stats_sticker

# Reference
# https://github.com/GuangchuangYu/hexSticker