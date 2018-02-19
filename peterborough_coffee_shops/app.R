library(shiny)
library(leaflet)
library(dplyr)
library(tibble)
library(sf)
library(stringr)

# https://emmavestesson.netlify.com/2018/02/what-should-i-have-for-lunch/

# cafes
cafe <- tribble(
  ~name,      ~long,       ~lat,
  "Becket Chapel", -0.2415962, 52.5727516,
  "Starbucks", -0.2421326,  52.572644,
  "Caffe Nero", -0.2421004, 52.5728885,
  "Argo Lounge", -0.2419395, 52.5722201,
  "Bewitched", -0.2420039, 52.5717898,
  "Costa Coffee", -0.2421165, 52.5715453,
  "Pret A Manger", -0.2423794,  52.573296,
  "Patisserie Valerie", -0.2429051, 52.5728396,
  "Costa Coffee", -0.2447397, 52.5735079,
  "John Lewis", -0.2449543, 52.5744729,
  "Marks & Spencer", -0.2445198, 52.5740719,
  "Westgate", -0.2430499, 52.5753434,
  "Market 1", -0.2391929, 52.5747955,
  "Market_2", -0.2395222, 52.5749454,
  "Nata", -0.2406728, 52.5740821,
  "Rivergate", -0.2428923,  52.569892,
  "Squires", -0.2459715, 52.5719277,
  "Waitrose", -0.2499466, 52.5753281,
  "Great Northern Hotel", -0.2491955, 52.5746503,
  "Pumpkin Cafe", -0.2497508, 52.5745765
) %>% 
  st_as_sf(coords = c("long", "lat"), crs = 4326)

#office
office <- tribble(
  ~name,      ~long,       ~lat,
  "office", -0.2388469, 52.5741681
) %>% 
  st_as_sf(coords = c("long", "lat"), crs = 4326)

# distance
dist <- st_distance(office, cafe)
cafe <- cafe %>% 
  mutate(distance = round(dist),
         label = str_c("<b>", name, "</b> <br />",
                       "Distance: ", distance, "m"))
max_distance <- unclass(max(cafe$distance))
min_distance <- unclass(min(cafe$distance))
# round to nearest 100
max_distance <- max_distance + (100 - max_distance %% 100)
min_distance <- min_distance + (100 - min_distance %% 100)

# UI        
ui <- shinyUI(fluidPage(
  
  sidebarPanel(width = 3,
               
    titlePanel("CoffeeR"),
    
    sliderInput("distance", "Select walking distance",
                min = min_distance, max = max_distance, 
                value = c(0, 545), 
                step = 100,
                post = "m"),
               
    actionButton("recalc", "Choose a new coffee suggestion"),
    h6(uiOutput("link"))
  ),
  mainPanel(width = 9,
            
    leafletOutput("mymap", height = 600),
    h3(textOutput("selected_var"))
            
  )
 )
)

# Server 
server <- function(input, output, session) {
  
  points <- eventReactive(input$recalc, {
    
    cafe <- cafe %>% 
      filter(unclass(distance) >= input$distance[1], 
             unclass(distance) <= input$distance[2]) %>% 
      sample_n(1)
    #sample_n(cafe, 1)
  }, ignoreNULL = FALSE)
  
  
  # create the interactive map...
  output$mymap <- renderLeaflet({
    leaflet(padding = 0, options= leafletOptions(minZoom = 10, maxZoom = 18)) %>% 
      addTiles()  %>%
      addMarkers(group = "The office",
                lng = -0.2388469,
                lat = 52.5741681, 
                popup = "The office") %>% 
      addCircleMarkers(group = "All cafe places",
                      lng = st_coordinates(cafe)[, 1],
                      lat = st_coordinates(cafe)[, 2],
                      radius = 8, weight = 0.25,
                      stroke = TRUE, opacity = 75,
                      fill = TRUE, fillColor = "deeppink",
                      fillOpacity = 100,
                      popup = cafe$label,
                      color = "white") %>%
      addCircleMarkers(data = points(), group = "Random coffee place", 
                       radius = 8, weight = 0.25,
                       stroke = TRUE, opacity = 100,
                       fill = TRUE, fillColor = "yellow",
                       fillOpacity = 100,     
                       popup = points()$label,
                       color = "white")
  })
  
  output$selected_var <- renderText({ 
    str_c("Today's suggestion for coffee is ",points()$name, 
          ". It is ", points()$distance, 
          "m walk away.") 
  })
  

  url <- a("What should I have for lunch?", 
           href="https://emmavestesson.netlify.com/2018/02/what-should-i-have-for-lunch/")
  output$link <- renderUI({
    tagList("Adapted from ", url)
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
