# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# published on: 20 June 2020

# Custom Functions are the functions written for dashboard in such a way
# that team is not voiding the DRY principle. These functions are imported 
# in the server logic of the dashboard.

#' function for issue plotting over years
#'
#' @param result a dataframe filtered based on global filters of ministries
#' @param token1 From token based on Markov Chain plot
#' @param token2 To token based on Markov Chain plot
#'
#' @return plot generated for given tokens from Entity Analysis
#' @export
#'
issue_plot <- function(result, token1, token2) {
  agg <-
    result %>% group_by(Year) %>% summarise(count = n()) %>%
    mutate(per = paste(round(count / sum(count), 2) * 100, '%'))
  
  cw1 <- capitalize(token1)
  cw2 <- capitalize(token2)
  
  title <-
    bquote(paste(bold(.(cw1)) ~ bold(.(cw2)) ~ "Concern Over Years"))
  
  ggplot(agg, aes(x = as.factor(Year), y = per)) +
    geom_bar(stat = 'identity' , fill = "steelblue") +
    geom_text(aes(label = per),
              vjust = 1.6,
              color = "white",
              size = 5) +
    ggtitle(label = title) +
    ylab(label = 'Responses') +
    xlab(label = 'Years') +
    theme_minimal()
}

#' function for sentiment highlight
#'
#' @param result filtered dataframe based on the tokens from Markov Chain plot and global ministry filters 
#' @param file_name name of the output file
#'
#' @return create html file with highlighted feedbacks with red and green colors
#' @export
#'
plot_sentiment <- function(result, file_name) {
  df <- result
  
  feedbacks <-
    df %>% select(Comment) %>% mutate(id = row_number())
  group_df <-
    feedbacks %>% get_sentences(feedbacks$Comment) %>% select(Comment, id)
  
  sent_df <-
    sentiment(df$Comment) %>% select(element_id, sentiment)
  
  highlight_df <-
    cbind(group_df, sent_df) %>% select(sentiment, Comment, id)
  
  highlight_df %>% mutate(review = get_sentences(Comment)) %$% sentiment_by(review, id) %>%
    highlight(file = file_name, open = FALSE)
}

#' function for comparison plots for Themes
#' @param value frequency of the labels
#' @param total_comments number of comments in the datafarme
#'
#' @return percentage value for labels
#' @export
#'
high_bar <- function(value, total_comments) {
  value * 100 / total_comments
}

#' function to plot theme comparison charts for comparison tab 
#'
#' @param datatable a dataframe
#' @param total_comments  total number of comments to compute percentage
#' @param title title of the plot
#'
#' @return plots for Concerns/Appreciations
#' @export
#'
label_bar <- function(datatable, total_comments, title) {
  ggplot(data = datatable, aes(
    x = key,
    y = high_bar(value, total_comments),
    fill = key
  )) +
    geom_bar(
      stat = "identity",
      show.legend = FALSE,
      fill = "skyblue4",
      color = "black",
      width = 0.8
    ) +
    geom_text(aes(
      label = round(high_bar(value, total_comments), 1),
      y = high_bar(value, total_comments) + 1
    ),
    color = "gray40") +
    labs(title = title, x = 'Themes', y = 'Percentage of comments (%)') +
    theme_bw()
}

#' function for trend lines
#'
#' @param data_t dataframe for trend lines
#'
#' @return plot with trend lines for Concerns/Appreciations for a selected label
#' @export
#'
plot_data <- function(data_t) {
  data_t %>% ggplot(aes(
    x = Year,
    y = comments_per,
    group = as.factor(Type)
  )) +
    geom_line(aes(color = as.factor(Type))) +
    geom_point(aes(color = as.factor(Type))) +
    labs(color = 'Type', x = 'Years', y = 'Percentage of comments (%)') +
    theme_minimal()
}


#' function for getting data ready to plot trend chart
#'
#' @param data a dataframe
#' @param sel_column label selected from UI to plot trend line for
#'
#' @return plot with a trend line for selected label
#' @export
#'
plot_trend <- function(data, sel_column) {
  datatable_num <- data %>%
    group_by(Year, Question) %>%
    summarise(comments = sum(!!sel_column)) %>%
    rename(Year_1 = Year,
           Question_1 = Question)
  
  datatable_den <- data %>%
    group_by(Year, Question) %>%
    summarise(counts = n())
  
  datatable <- cbind(datatable_den, datatable_num)
  
  datatable <- datatable %>%
    select(Year, Question, counts, comments) %>%
    mutate(comments_per = round((comments / counts) * 100), 2) %>%
    mutate(Type = ifelse(Question == 1, 'Concerns', 'Appreciations'))
  
  plot_data(datatable)
}

