# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# published on: 20 June 2020

# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.

library("shiny")
library("shinydashboard")
library("RColorBrewer")
library("shinycssloaders")
library("shinyBS")
library("tidyverse")

# UI for Dashboard Code that creates and handles UI elements of dashboard
ui <- dashboardPage(
    # header
    dashboardHeader(title = 'BC Stats - Text Analytics', titleWidth = '15%'),
    
    #sidebar content
    dashboardSidebar(
        sidebarMenu(
            id = 'tab',
            menuItem('Concerns',
                     tabName = 'q1',
                     icon =  icon('chart-bar')),
            menuItem(
                'Appreciations',
                tabName = 'q2',
                icon = icon('chart-bar')
            ),
            menuItem(
                'Comparison',
                tabName = 'comparison',
                icon = icon('project-diagram')
            ),
            menuItem(
                'Data Dictionary',
                tabName = 'data_dictionary',
                icon = icon('book-open')
            )
        ),
        collapsed = TRUE
    ),
    
    
    
    #body content
    dashboardBody(
        tags$script(HTML("$('body').addClass('fixed');")),
        
        tabItems(
            # 1st tab
            tabItem(
                tabName = 'q1',
                titlePanel(title = 'Concerns'),
                
                # ministry selector
                fluidRow(box(
                    title = 'Select Ministries',
                    selectizeInput(
                        'ministry_names',
                        label = NULL,
                        choices = NULL,
                        multiple = TRUE,
                        options = list(create = TRUE),
                    ),
                    width = 12
                )),
                
                fluidRow(
                    #Total Records
                    valueBoxOutput('total_records', width = 4)  %>% withSpinner(color =
                                                                                    "skyblue"),
                    # 2013
                    valueBoxOutput('records_2k13', width = 2),
                    
                    # 2015
                    valueBoxOutput('records_2k15', width = 2),
                    
                    # 2018
                    valueBoxOutput('records_2k18', width = 2),
                    
                    # 2020
                    valueBoxOutput('records_2k20', width = 2)
                ),
                
                fluidRow(
                    box(
                        title = 'Employee Concerns',
                        plotOutput('plot_wc') %>% withSpinner(color = "skyblue")
                    ),
                    box(
                        title = 'Polarity',
                        plotOutput('plot_pn') %>% withSpinner(color = "skyblue")
                    )
                ),
                
                fluidRow(
                    box(
                        title = 'Markov Chain Text Processing',
                        plotOutput('plot_mc') %>% withSpinner(color = "skyblue"),
                        width = 8,
                        height = 550
                    ),
                    box(
                        title = 'Markov Threshold',
                        sliderInput("slider_mc", "Minimum Occurrences:", 1, 600, 60),
                        box(
                            title = 'Entity Analysis',
                            collapsible = TRUE,
                            collapsed = TRUE,
                            status = 'primary',
                            solidHeader = TRUE,
                            textInput(
                                "text_word1",
                                label = h5("From Token"),
                                value = '',
                                width = '50%'
                            ),
                            textInput(
                                "text_word2",
                                label = h5("To Token"),
                                value = '',
                                width = '50%'
                            ),
                            checkboxGroupInput(
                                "checkGroup_ea",
                                label = h3("Mining Options"),
                                choices = list(
                                    "Issues Over Time" = 1,
                                    "Sentiment Highlights" = 2
                                ),
                                selected = ''
                            ),
                            width = 12
                        )
                        ,
                        width = 4
                    )
                ),
                
                
                fluidRow(
                    class = "flex-nowrap",
                    
                    box(
                        title = 'Issues Over Years',
                        plotOutput('plot_issue') %>% withSpinner(color = "skyblue"),
                        width = 4,
                        collapsible = TRUE,
                        collapsed = TRUE
                    ),
                    
                    box(
                        title = 'Sentiment Analysis',
                        htmlOutput('sentiment'),
                        width = 8,
                        collapsible = TRUE,
                        collapsed = TRUE
                    )
                )
            ),
            
            
            # 2nd tab
            tabItem(
                tabName = 'q2',
                titlePanel(title = 'Appreciations'),
                fluidRow(# ministry selector
                    box(
                        title = 'Select Ministries',
                        selectizeInput(
                            'ministry_names_q2',
                            label = NULL,
                            choices = NULL,
                            multiple = TRUE,
                            options = list(create = TRUE),
                        ),
                        width = 12
                    )),
                
                fluidRow(
                    #Total Records
                    valueBoxOutput('total_records_q2', width = 3)  %>% withSpinner(color =
                                                                                       "skyblue"),
                    
                    # 2015
                    valueBoxOutput('records_2k15_q2', width = 3),
                    
                    # 2018
                    valueBoxOutput('records_2k18_q2', width = 3),
                    
                    # 2020
                    valueBoxOutput('records_2k20_q2', width = 3)
                ),
                
                fluidRow(
                    box(
                        title = 'Employee Concerns',
                        plotOutput('plot_wc_q2') %>% withSpinner(color = "skyblue")
                    ),
                    box(
                        title = 'Polarity',
                        plotOutput('plot_pn_q2') %>% withSpinner(color = "skyblue")
                    )
                ),
                
                fluidRow(
                    box(
                        title = 'Markov Chain Text Processing',
                        plotOutput('plot_mc_q2') %>% withSpinner(color = "skyblue"),
                        width = 8,
                        height = 550
                    ),
                    box(
                        title = 'Markov Threshold',
                        sliderInput("slider_mc_q2", "Minimum Occurrences:", 1, 600, 25),
                        box(
                            title = 'Entity Analysis',
                            collapsible = TRUE,
                            collapsed = TRUE,
                            status = 'primary',
                            solidHeader = TRUE,
                            textInput(
                                "text_word1_q2",
                                label = h5("From Token"),
                                value = '',
                                width = '50%'
                            ),
                            textInput(
                                "text_word2_q2",
                                label = h5("To Token"),
                                value = '',
                                width = '50%'
                            ),
                            checkboxGroupInput(
                                "checkGroup_ea_q2",
                                label = h3("Mining Options"),
                                choices = list(
                                    "Issues Over Time" = 1,
                                    "Sentiment Highlights" = 2
                                ),
                                selected = ''
                            ),
                            width = 12
                        )
                        ,
                        width = 4
                    )
                ),
                
                
                fluidRow(
                    class = "flex-nowrap",
                    
                    box(
                        title = 'Issues Over Years',
                        plotOutput('plot_issue_q2') %>% withSpinner(color = "skyblue"),
                        width = 4,
                        collapsible = TRUE,
                        collapsed = TRUE
                    ),
                    
                    box(
                        title = 'Sentiment Analysis',
                        htmlOutput('sentiment_q2'),
                        width = 8,
                        collapsible = TRUE,
                        collapsed = TRUE
                    )
                )
            ),
            
            # comparison tab
            tabItem(
                tabName = 'comparison',
                titlePanel(title = 'Comparison'),
                
                # ministry selector
                fluidRow(box(
                    title = 'Select Ministries',
                    selectizeInput(
                        'ministry_names_comparison',
                        label = NULL,
                        choices = NULL,
                        multiple = TRUE,
                        options = list(create = TRUE),
                    ),
                    width = 12
                )),
                
                fluidRow(
                    box(
                        title = 'Themes - Concerns',
                        width = 5,
                        plotOutput('plot_themes_q1') %>% withSpinner(color = "skyblue")
                    ),
                    box(
                        title = 'Themes - Appreciations',
                        width = 5,
                        plotOutput('plot_themes_q2') %>% withSpinner(color = "skyblue")
                    ),
                    box(
                        title = 'Pick Year',
                        width = 2,
                        selectizeInput(
                            'pick_year',
                            label = NULL,
                            choices = NULL,
                            multiple = TRUE,
                            options = list(create = TRUE),
                        )
                    )
                ),
                
                fluidRow(
                    
                    box(
                        title = 'Trend',
                        width = 10,
                        footer = 'NOTE: Labels used in graphs for question 2 are predictions from Bi-GRU.',
                        plotOutput('plot_trend') %>% withSpinner(color = "skyblue")
                    ),
                    box(
                        title = 'Pick Label',
                        width = 2,
                        selectizeInput(
                            'pick_label',
                            label = NULL,
                            choices = NULL,
                            multiple = FALSE,
                            options = list(create = TRUE),
                            
                        )
                    )
                )
                
            ),
            
            # data dictionary tab
            tabItem(tabName = 'data_dictionary',
                    fluidRow(
                        box(
                            title = 'Metadata',
                            includeMarkdown('data_dictionary.md'),
                            width = 12,
                            status = 'info',
                            solidHeader = TRUE,
                            footer = 'Note: Themes and Sub-Themes are subject to change with new data. '
                        )
                    ))
        )
    )
)

