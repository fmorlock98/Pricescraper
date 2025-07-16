# MA Pricescraper

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This master’s thesis investigates the cost-benefit relationship of heat pumps in Europe, focusing on whether investing in higher energy efficiency is financially worthwhile for residential consumers. As heat pumps become a key technology in Europe’s transition away from fossil fuels, the study examines how increased efficiency—measured by performance metrics like SCOP—affects both initial purchase prices and long-term operational savings. Using data from the Keymark database and scraped online prices, the project applies statistical analysis and visualization techniques to identify trends in pricing and efficiency. The results aim to guide consumers, manufacturers, and policymakers on the economic trade-offs of heat pump efficiency, with the goal of supporting more informed purchasing decisions and effective policy design.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── Pricescraper   <- Source code. Execute the scripts in the order of their numbering.
    │
    ├── 2_scrapers            <- Webscraper for the 14 selected websites.
    │
    ├── 4_matching               <- Matching scripts for 13 selected manufacturers.
    │
    ├── 6_analysis              <- Data analysis scripts.
    │
    ├── 1_filter_hplib.py             <- Python script to filter the original hplib.
    │
    ├── 3_combine_and_standardize_scraped_data.py             <- Python file to organize the  │                                                            scraped data.
    │
    └── 5_combine_check_decoding_matched_manufacturers.py             <- Python script to   generate the final database for analysis.
```

--------

