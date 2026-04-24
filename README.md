# mental_health_ml


# Dataset
DEFRA's Automatic Urban and Rural Network [(AURN dataset)](https://uk-air.defra.gov.uk/networks/network-info?view=aurn).

## Monitoring Sites
Lincoln Canwick Road
Immingham Woodlands Avenue
Scunthorpe Town Rowland Road
Toft Newton Reservior (near Market Rasen)
Tallington River Welland (near Stamford)

| Site Name (UK-AIR ID)           | DEFRA Environment Type | Category        | Pollutants (NO2, PM2.5, PM10) | Research Rationale |
|---------------------------------|--------------------------|-----------------------|-------------------------------|--------------------|
| Lincoln Canwick Road (UKA00561) | Urban Traffic            | Small Urban-Deprived  | NO2                           | High-density urban roadside traffic-heavy |
| Scunthorpe Town (UKA00381)      | Urban Industrial         | Small Urban-Deprived  | NO2, PM2.5, PM10              | Evaluating localised risk in an industrial-urban setting situated near heavy industry (steelworks). |
| Immingham Woodlands Avenue (UKA00647)  | Urban Background         | Coastal Industrial    | NO2, PM2.5, PM10              | Industrial-urban setting with persistent inequalities. |
| Toft Newton (UKA01026)          | Rural Background         | Rural                 | NO2, PM2.5, PM10              | Provides a rural baseline to evaluate system performance in clean air environments. |
| Tallington (UKA01038)           | Rural Background         | Rural                 | PM2.5, PM10                   | Analyses the impact of transport infrastructure on rural areas. |

## Pollutants
- Nitrogen dioxide NO2
- PM 2.5 (fine particulate matter with a diameter of less than 2.5 micrometres, 1/400th mm)
- PM 10 (fine particulate matter with a diameter of less than 10 micrometres)

In literature link between high levels of NO2, PM2.5 and PM 10 to mental health disorders:
- https://www.sciencedirect.com/science/article/pii/S016517811830800X
- https://www.cambridge.org/core/journals/the-british-journal-of-psychiatry/article/association-between-air-pollution-exposure-and-mental-health-service-use-among-individuals-with-first-presentations-of-psychotic-and-mood-disorders-retrospective-cohort-study/010F283B9107A5F04C51F90B5D5F96D6
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9812022/

## Machine Learning Model
