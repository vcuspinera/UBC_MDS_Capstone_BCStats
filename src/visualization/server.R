# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# published on: 20 June 2020

# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' from UI file.

# loading standard libraries
library("shiny")
library("shinydashboard")
library("wordcloud")
library("SnowballC")
library("RColorBrewer")
library("tm")
library("readxl")
library("tidytext")
library("textdata")
library("tidyverse")
library("tidyr")
library("tokenizers")
library("igraph")
library("ggraph")
library("shinycssloaders")
library("magrittr")
library("stringr")
library("data.table")
library("Hmisc")
library("sentimentr")
library("shinyBS")
library("rlang")

# loading our custom functions
source('custom_functions.R')

# Define server logic required to run dashboard
server <- function(input, output, comments, session) {
    # setting seed
    set.seed(1234)
    
    # loading raw data for Concerns
    data_df_q1 <-
        read_excel('../../data/interim/question1_models/advance/ministries_Q1_all.xlsx')
    
    # loading raw data for Appreciations
    data_df_q2 <-
        read_excel('../../data/interim/question2_models/ministries_Q2.xlsx')
    
    # loading data for Comparison
    data_df_comp <-
        read_excel('../../data/interim/question2_models/ministries_Q2_pred.xlsx')
    
    
    ## checking which tab is active
    observe({
        tabs <- input$tab
        
        if (input$tab == 'q1') {
            # creating new objects for transformations
            comments <- data_df_q1
            only_comments <- comments %>% select(Comment)
            
            # updating selector with ministry names
            ministries <-
                comments %>% select(Ministry) %>% filter(Ministry != 'NA') %>% unique()
            updateSelectizeInput(
                session,
                'ministry_names',
                choices = sort(ministries$Ministry),
                server = TRUE
            )
            
            # updating dashboard according to ministries
            observe({
                user_filter <- input$ministry_names
                
                if (length(user_filter) != 0) {
                    comments <- data_df_q1 %>% filter(Ministry %in% user_filter)
                    only_comments <-
                        comments %>% filter(Ministry %in% user_filter) %>% select(Comment)
                    smry <-
                        comments %>% filter(Ministry %in% user_filter) %>% group_by(Year) %>%
                        summarise(count = n())
                    
                }
                else{
                    comments <- data_df_q1
                    only_comments <- comments %>% select(Comment)
                    smry <-
                        comments %>% group_by(Year) %>% summarise(count = n())
                }
                
                #Updating Data Stats
                #Total Records
                output$total_records <- renderValueBox({
                    valueBox(
                        nrow(only_comments),
                        "Total Respondents",
                        icon = icon("poll"),
                        color = "purple"
                    )
                })
                
                
                # 2013 Records
                output$records_2k13 <- renderValueBox({
                    valueBox(smry$count[1],
                             "2013",
                             icon = icon("users"),
                             color = "orange")
                })
                
                # 2015 Records
                output$records_2k15 <- renderValueBox({
                    valueBox(smry$count[2],
                             "2015",
                             icon = icon("users"),
                             color = "orange")
                })
                
                # 2018 Records
                output$records_2k18 <- renderValueBox({
                    valueBox(smry$count[3],
                             "2018",
                             icon = icon("users"),
                             color = "yellow")
                })
                
                # 2020 Records
                output$records_2k20 <- renderValueBox({
                    valueBox(smry$count[4],
                             "2020",
                             icon = icon("users"),
                             color = "aqua")
                })
                
                
                # Text Mining
                single_tokens <-
                    only_comments %>% unnest_tokens(word, Comment)
                
                bing_word_counts <- single_tokens %>%
                    inner_join(get_sentiments("bing")) %>%
                    count(word, sentiment, sort = TRUE) %>%
                    ungroup()
                
                
                # Word Cloud Plot
                output$plot_wc <- renderPlot({
                    # VERSION - Tidy
                    plot <- single_tokens %>%
                        anti_join(stop_words) %>%
                        count(word) %>%
                        with(
                            wordcloud(
                                word,
                                n,
                                max.words = 100,
                                random.order = FALSE,
                                rot.per = 0.35,
                                colors = brewer.pal(8, "Dark2")
                            )
                        )
                    
                })
                
                # New Plot - Polarity
                output$plot_pn <- renderPlot({
                    bing_word_counts %>%
                        group_by(sentiment) %>%
                        top_n(10) %>%
                        ungroup() %>%
                        mutate(word = reorder(word, n)) %>%
                        ggplot(aes(word, n, fill = sentiment)) +
                        geom_col(show.legend = FALSE) +
                        facet_wrap(~ sentiment, scales = "free_y") +
                        labs(y = "Contribution to sentiment",
                             x = NULL) +
                        coord_flip() +
                        theme_minimal()
                })
                
                # New Plot - Markov Chain Text Processing
                bigrams <-
                    only_comments %>% mutate(line = row_number()) %>%
                    unnest_tokens(bigrams, Comment, token = 'ngrams', n = 2)
                
                bigrams_separated <- bigrams %>%
                    separate(bigrams, c("word1", "word2"), sep = " ")
                
                bigrams_filtered <- bigrams_separated %>%
                    filter(!word1 %in% stop_words$word) %>%
                    filter(!word2 %in% stop_words$word)
                
                # new bigram counts:
                bigram_counts <- bigrams_filtered %>%
                    count(word1, word2, sort = TRUE)
                
                bigram_counts <- bigram_counts %>% filter(!is.na(word1))
                
                # Plotting Markov Chains
                output$plot_mc <- renderPlot({
                    bigram_graph <- bigram_counts %>%
                        filter(n > input$slider_mc) %>%
                        graph_from_data_frame()
                    
                    a <-
                        grid::arrow(type = "closed", length = unit(.15, "inches"))
                    
                    ggraph(bigram_graph, layout = "fr") +
                        geom_edge_link(
                            aes(edge_alpha = n),
                            show.legend = FALSE,
                            arrow = a,
                            end_cap = circle(.07, 'inches')
                        ) +
                        geom_node_point(color = "lightblue", size = 6) +
                        geom_node_text(aes(label = name),
                                       vjust = 1,
                                       hjust = 1) +
                        theme_void()
                })
                
                # Entity Analysis - Option Checking
                observeEvent(input$checkGroup_ea, {
                    len <- length(input$checkGroup_ea)
                    val <- input$checkGroup_ea
                    
                    token1 <- input$text_word1
                    token2 <- input$text_word2
                    search_word <- paste(token1, token2)
                    result <-
                        comments[comments$Comment %like% search_word,] %>% select(Comment, Year)
                    
                    # Plotting Issues & Highlights
                    if (len == 2) {
                        # issues
                        output$plot_issue <- renderPlot({
                            issue_plot(result, token1, token2)
                            
                        })
                        
                        # sentiment
                        plot_sentiment(result, 'highlight.html')
                        
                        getPage <- function() {
                            return(includeHTML("highlight.html"))
                            #return(includeHTML("dummy.html")) # for dummy data
                        }
                        
                        output$sentiment <- renderUI({
                            #getPage()
                            tags$iframe(
                                srcdoc = paste(readLines('highlight.html'), collapse = '\n'),
                                #srcdoc = paste(readLines('dummy.html'), collapse = '\n'),
                                width = "100%",
                                height = "600px",
                                frameborder = "0"
                            )
                        })
                    }
                    
                    else if (len == 1 & val == 1) {
                        output$plot_issue <- renderPlot({
                            issue_plot(result, token1, token2)
                            
                        })
                    }
                    
                    else if (len == 1 & val == 2) {
                        plot_sentiment(result, 'highlight.html')
                        
                        getPage <- function() {
                            return(includeHTML("highlight.html"))
                            #return(includeHTML("dummy.html"))
                        }
                        
                        output$sentiment <- renderUI({
                            #getPage()
                            tags$iframe(
                                srcdoc = paste(readLines('highlight.html'), collapse = '\n'),
                                #srcdoc = paste(readLines('dummy.html'), collapse = '\n'),
                                width = "100%",
                                height = "600px",
                                frameborder = "0"
                            )
                        })
                    }
                    
                })
                
            }) # updating dashboard code ends here
            
        }
        else if (input$tab == 'q2') {
            # creating new objects for transformations
            comments <- data_df_q2
            only_comments <- comments %>% select(Comment)
            
            # updating selector with ministry names
            ministries <-
                comments %>% select(Ministry) %>% filter(Ministry != 'NA') %>% unique()
            updateSelectizeInput(
                session,
                'ministry_names_q2',
                choices = sort(ministries$Ministry),
                server = TRUE
            )
            
            # updating dashboard according to ministries
            observe({
                user_filter <- input$ministry_names_q2
                
                if (length(user_filter) != 0) {
                    comments <- data_df_q2 %>% filter(Ministry %in% user_filter)
                    only_comments <-
                        comments %>% filter(Ministry %in% user_filter) %>% select(Comment)
                    smry <-
                        comments %>% filter(Ministry %in% user_filter) %>% group_by(Year) %>% summarise(count = n())
                    
                }
                else{
                    comments <- data_df_q2
                    only_comments <- comments %>% select(Comment)
                    smry <-
                        comments %>% group_by(Year) %>% summarise(count = n())
                }
                
                #Updating Data Stats
                #Total Records
                output$total_records_q2 <- renderValueBox({
                    valueBox(
                        nrow(only_comments),
                        "Total Respondents",
                        icon = icon("poll"),
                        color = "purple"
                    )
                })
                
                
                # 2015 Records
                output$records_2k15_q2 <- renderValueBox({
                    valueBox(smry$count[1],
                             "2015",
                             icon = icon("users"),
                             color = "orange")
                })
                
                # 2018 Records
                output$records_2k18_q2 <- renderValueBox({
                    valueBox(smry$count[2],
                             "2018",
                             icon = icon("users"),
                             color = "yellow")
                })
                
                # 2020 Records
                output$records_2k20_q2 <- renderValueBox({
                    valueBox(smry$count[3],
                             "2020",
                             icon = icon("users"),
                             color = "aqua")
                })
                
                
                # Text Mining
                single_tokens <-
                    only_comments %>% unnest_tokens(word, Comment)
                
                bing_word_counts <- single_tokens %>%
                    inner_join(get_sentiments("bing")) %>%
                    count(word, sentiment, sort = TRUE) %>%
                    ungroup()
                
                
                # Word Cloud Plot
                output$plot_wc_q2 <- renderPlot({
                    # VERSION - Tidy
                    plot <- single_tokens %>%
                        anti_join(stop_words) %>%
                        count(word) %>%
                        with(
                            wordcloud(
                                word,
                                n,
                                max.words = 100,
                                random.order = FALSE,
                                rot.per = 0.35,
                                colors = brewer.pal(8, "Dark2")
                            )
                        )
                    
                })
                
                # New Plot - Polarity
                output$plot_pn_q2 <- renderPlot({
                    bing_word_counts %>%
                        group_by(sentiment) %>%
                        top_n(10) %>%
                        ungroup() %>%
                        mutate(word = reorder(word, n)) %>%
                        ggplot(aes(word, n, fill = sentiment)) +
                        geom_col(show.legend = FALSE) +
                        facet_wrap(~ sentiment, scales = "free_y") +
                        labs(y = "Contribution to sentiment",
                             x = NULL) +
                        coord_flip() +
                        theme_minimal()
                })
                
                # New Plot - Markov Chain Text Processing
                bigrams <-
                    only_comments %>% mutate(line = row_number()) %>%
                    unnest_tokens(bigrams, Comment, token = 'ngrams', n = 2)
                
                bigrams_separated <- bigrams %>%
                    separate(bigrams, c("word1", "word2"), sep = " ")
                
                bigrams_filtered <- bigrams_separated %>%
                    filter(!word1 %in% stop_words$word) %>%
                    filter(!word2 %in% stop_words$word)
                
                # new bigram counts:
                bigram_counts <- bigrams_filtered %>%
                    count(word1, word2, sort = TRUE)
                
                bigram_counts <- bigram_counts %>% filter(!is.na(word1))
                
                # Plotting Markov Chains
                output$plot_mc_q2 <- renderPlot({
                    bigram_graph <- bigram_counts %>%
                        filter(n > input$slider_mc_q2) %>%
                        graph_from_data_frame()
                    
                    a <-
                        grid::arrow(type = "closed", length = unit(.15, "inches"))
                    
                    ggraph(bigram_graph, layout = "fr") +
                        geom_edge_link(
                            aes(edge_alpha = n),
                            show.legend = FALSE,
                            arrow = a,
                            end_cap = circle(.07, 'inches')
                        ) +
                        geom_node_point(color = "lightblue", size = 6) +
                        geom_node_text(aes(label = name),
                                       vjust = 1,
                                       hjust = 1) +
                        theme_void()
                })
                
                # Entity Analysis - Option Checking
                observeEvent(input$checkGroup_ea_q2, {
                    len <- length(input$checkGroup_ea_q2)
                    val <- input$checkGroup_ea_q2
                    
                    token1 <- input$text_word1_q2
                    token2 <- input$text_word2_q2
                    search_word <- paste(token1, token2)
                    result <-
                        comments[comments$Comment %like% search_word,] %>% select(Comment, Year)
                    
                    # Plotting Issues & Highlights
                    if (len == 2) {
                        # issues
                        output$plot_issue_q2 <- renderPlot({
                            issue_plot(result, token1, token2)
                            
                        })
                        
                        # sentiment
                        plot_sentiment(result, 'highlight_q2.html')
                        
                        getPage <- function() {
                            return(includeHTML("highlight_q2.html"))
                            #return(includeHTML("dummy.html"))
                        }
                        
                        output$sentiment_q2 <- renderUI({
                            #getPage()
                            tags$iframe(
                                srcdoc = paste(readLines('highlight_q2.html'), collapse = '\n'),
                                #srcdoc = paste(readLines('dummy.html'), collapse = '\n'),
                                width = "100%",
                                height = "600px",
                                frameborder = "0"
                            )
                        })
                    }
                    
                    else if (len == 1 & val == 1) {
                        output$plot_issue_q2 <- renderPlot({
                            issue_plot(result, token1, token2)
                            
                        })
                    }
                    
                    else if (len == 1 & val == 2) {
                        plot_sentiment(result, 'highlight_q2.html')
                        
                        getPage <- function() {
                            return(includeHTML("highlight_q2.html"))
                            #return(includeHTML("dummy.html"))
                        }
                        
                        output$sentiment_q2 <- renderUI({
                            #getPage()
                            tags$iframe(
                                srcdoc = paste(readLines('highlight_q2.html'), collapse = '\n'),
                                #srcdoc = paste(readLines('dummy.html'), collapse = '\n'),
                                width = "100%",
                                height = "600px",
                                frameborder = "0"
                            )
                        })
                    }
                    
                })
                
            }) # updating dashboard code ends here for second tab
        }
        
        else if (input$tab == 'comparison') {
            # creating new objects for transformations
            comments_q1 <- data_df_q1
            
            comments_q2 <- data_df_comp
            
            
            # updating selector with ministry names
            ministries <-
                comments_q1 %>% select(Ministry) %>% filter(Ministry != 'NA') %>% unique()
            updateSelectizeInput(
                session,
                'ministry_names_comparison',
                choices = sort(ministries$Ministry),
                server = TRUE
            )
            
            
            # prepare data for trends
            trends_df <- data_df_comp
            
            ### Data Preparation Question 1
            q1_subset <- data_df_q1
            q1_subset['Question'] = 1
            
            ### Question 2 cleaning and data formatting
            # comments with atleast one assigned labels
            rows_to_keep <- rowSums(trends_df[, c(6:17)]) != 0
            
            # filtering data
            trends_df <- trends_df[rows_to_keep, ]
            trends_df['Question'] = 2
            trend_data <-
                rbind(trends_df, q1_subset) # creating a unified dataframe
            
            
            # observing events as per selected ministry for comparison
            observe({
                user_filter_comparison <- input$ministry_names_comparison
                
                # updating data
                if (length(user_filter_comparison) != 0) {
                    # for question1
                    comments_q1 <-
                        data_df_q1 %>% filter(Ministry %in% user_filter_comparison)
                    
                    # for question 2
                    comments_q2 <-
                        data_df_comp %>% filter(Ministry %in% user_filter_comparison)
                    
                    # for trend line plot
                    comments_trends <-
                        trend_data %>% filter(Ministry %in% user_filter_comparison)
                    
                }
                else{
                    # for question 1
                    comments_q1 <- data_df_q1
                    
                    # for question 2
                    comments_q2 <- data_df_comp
                    
                    # for trend
                    comments_trends <- trend_data
                }
                
                
                # get unique years and update the filter
                years <-
                    comments_q1 %>% select(Year)  %>% unique()
                
                updateSelectizeInput(session,
                                     'pick_year',
                                     choices = sort(years$Year),
                                     server = TRUE)
                
                # get unique labels and update the filter
                labels <- colnames(comments_q1)
                
                updateSelectizeInput(
                    session,
                    'pick_label',
                    choices = labels[6:17],
                    selected = labels[6],
                    server = TRUE
                )
                
                
                # observing year for updating themes plots
                observe({
                    user_filter_year <- input$pick_year
                    
                    # updating data
                    if (length(user_filter_year) != 0) {
                        # question 1
                        comments_q1_year <-
                            comments_q1 %>% filter(Year %in% user_filter_year)
                        
                        # question 2
                        comments_q2_year <-
                            comments_q2 %>% filter(Year %in% user_filter_year)
                        
                    }
                    else{
                        # for question 1
                        comments_q1_year <- comments_q1
                        
                        # for question 2
                        comments_q2_year <- comments_q2
                    }
                    
                    ### Question 1 cleaning and data formatting
                    sumdata_q1 <-
                        data.frame(value = apply(comments_q1_year[, c(6:17)], 2, sum))
                    sumdata_q1$key = rownames(sumdata_q1)
                    
                    # plots by Theme
                    total_comments_q1 <- dim(comments_q1_year)[1]
                    
                    output$plot_themes_q1 <- renderPlot({
                        label_bar(sumdata_q1, total_comments_q1, title = "Theme distribution across comments conveying concerns")
                    })
                    
                    
                    ### Question 2 cleaning and data formatting
                    # comments with atleast one assigned labels
                    rows_to_keep <- rowSums(comments_q2_year[, c(6:17)]) != 0
                    
                    # filtering data
                    data_df_q2 <- comments_q2_year[rows_to_keep, ]
                    
                    # transforming data for graph, by theme
                    sumdata_q2 <-
                        data.frame(value = apply(data_df_q2[, c(6:17)], 2, sum))
                    sumdata_q2$key = rownames(sumdata_q2)
                    
                    total_comments_q2 <- dim(data_df_q2)[1]
                    
                    output$plot_themes_q2 <- renderPlot({
                        label_bar(
                            datatable = sumdata_q2,
                            total_comments = total_comments_q2,
                            title = "Theme distribution across comments conveying improvements"
                        )
                    })
                    
                    
                })
                
                # observing Labels filter to have refreshed trend plot
                observe({
                    user_filter_label <- input$pick_label
                    
                    # updating data
                    if (user_filter_label != '') {
                        if (length(user_filter_label) != 0) {
                            # trends
                            selected_columns <-
                                comments_trends %>% select(Year, Question, user_filter_label)
                            
                            output$plot_trend <- renderPlot({
                                plot_trend(selected_columns, sym(user_filter_label))
                            })
                        }
                        
                    } # trend plot updating code
                    
                })
                
            }) # observing end for comparison
            
        }
        
    })
    
    
    
}
