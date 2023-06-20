# DHH23 Portfolio

This is a repository for my individual portfolio for the DHH23 Early Modern work. The aim of this project is to explore reuse in illustrations.


## Data

Data is from [Early Modern sample](https://github.com/dhh23/early_modern_data#early_modern_data)

## Directory structure

The project repository structured is variation of formats laid out in a few data science project organization articles (see the end of this README). `code` and `output` -directories include `work/` and `final/` -subdirectories. The `work/` -subdirectory is optional, but helps to keep development material separate from the polished and clean end products that should reside in the `final/` directory If the project only has "final" products, that directory level can of course be omitted too. 

````
project_name/
├── README.md              # project overview
├── documentation/         # project documentation
├── data/
│   ├── raw/               # immutable raw input data
│   ├── work/              # intermediate data
│   └── final/             # processed data for final analysis tasks
├── code/
│   ├── templates/         # code templates shared across projects. Visualization styles etc.
│   ├── work/
│   │   ├── person1/       # a directory for each person or task
│   │   ├── person2/        
│   │   └── task1/         
│   └── final/
│       ├── task1/         # A directory for each analysis task
│       └── another_task/  # if needed.
└── output/
    ├── figures/
    │   ├── work/
    │   └── final/
    └── publications/
        ├── work/
        └── final/
````

### Logic

* **[documentation/]:** Project meta documentation. Links to all relevant planning papers, interim notes, google drive folders, etc.
* **[code/]:** Data processing code. Finished code used for publication should be moved to *[final/]* subdirectory. Organization of the development directory *[work/]* can vary and the breakdown by person or task is just a suggestion. All directories, but especially *[final/]* should include a `README.md` clearly documenting what each script does.
    * `[code/templates/]` has templates for code shared across projects, such as visualization styles.
* **[output/]:** Both figures and publication texts/files. Divided to work and final subdirectories.

## References

* [PLoS Comput Biol. 2016 Jul; 12(7): **Ten Simple Rules for Taking Advantage of Git and GitHub**. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4945047/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4945047/) 
* [Human in a Machine World. May 25, 2016: **Folder Structure for Data Analysis**. https://medium.com/human-in-a-machine-world/folder-structure-for-data-analysis-62a84949a6ce](https://medium.com/human-in-a-machine-world/folder-structure-for-data-analysis-62a84949a6ce)
* [**Cookiecutter Data Science** - A logical, reasonably standardized, but flexible project structure for doing and sharing data science work. https://drivendata.github.io/cookiecutter-data-science/](https://drivendata.github.io/cookiecutter-data-science/)
* [Thinking on Data. December 9, 2018: **Best practices organizing data science projects**. https://www.thinkingondata.com/how-to-organize-data-science-projects/](https://www.thinkingondata.com/how-to-organize-data-science-projects/)
