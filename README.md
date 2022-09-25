# Project name
## OUR PLAN

Our plan is to develop a tool that uses a link to a twitter profile as an input to automatically scrape data of interactions between users and general information on their twitter profiles. This data is then used to create interactive visualizations and tables containing descriptive statistics and other information about the network, which are automatically saved on the users machine.

We plan on making the tool modular in a way that it can easily be extended with additional features in the future. The modularity would also allow researchers to not use the tool as a 'one-stop-shop', but also only use parts of the tool, such as either the scraping or the analysis seperately.

## TODO

- add Caveat that network will be plotted in a not fully connected way since we only count two way interactions

## Future Ideas

- have a full report combined on one webpage that can be saved to pdf
https://stackoverflow.com/questions/36262748/save-plotly-plot-to-local-file-and-insert-into-html
- cache users for faster scraping 


## Members

JH https://github.com/digital-scrappy

TD https://github.com/timo-damm
## Dependencies
R
Python
Pandoc
gcc-fortan
python dependencies in requirements.txt
R dependencies specified in R script
## Tool Description
In the [Bellingcat Survey](https://www.bellingcat.com/resources/2022/08/12/these-are-the-tools-open-source-researchers-say-they-need/) a majority researchers indicated they are in need of *free* tools that are *easy to use* (i.e. do not include writing your own code, using GitHub, or extensive use of the command line. With regards to network analysis specifically, users were in need of "*a network analysis tool to analyse social media connections*", which includes measures such as degree centrality and displays clear graphs. Other users were in need of "*a simple tool to visualize connections*" or "*a tool that helps draw network diagrams from data*"
The tool we developed produces easily understandable graphs and information about the network, without having to have knowledge on how to write code or use a command line interface. 
In it's most simple form, the user has to use the command line only to download the tool and give it a link to a twitter profile as an input. The tool then automatically uses snscrape to scrape the profile's data, such as who their user name, follower count, their friends count, their interactions, media uploads, retweets etc. 
Based on the initial profile, the tool then assembles an edgelist which contains every interaction between this profile and other users, as well as the interactions two users that interacted with the original profile among each other. The edgelist is the primary data containing information about the network. 
Other information, such as follower count, friends count and years active or frequency of activity on twitter is assembled into a separate dataset.

Using the ```igraph``` package in R, the tool then adds nodal characteristics from the separate dataset to the edgelist after transforming it into a network object. The resulting information-rich network is then used to create interactive visualizations of the network using the ```plotly``` package for R, descriptive statistics and other ineractive graphs using the ```plotly``` package, including the accounts with the most followers and posts. The descriptive statistics are adapted from Gerber and colleagues' (2013) reporting framework for network analyses (see [here](https://onlinelibrary.wiley.com/doi/pdf/10.1111/ajps.12011?casa_token=MTVxax7BWfkAAAAA:e6v3H2ciJlZT1BRuF1vauHmeuJnnGLjarp91CNuY2RaDMCC1x-awCF6iVQAtBLIr655VGFGXGyocXkBZ)).
The interactive visualizations (both the network graph and the other graphs) are saved as ready-to-use .html files in a designated folder on the user's machine. The table with descriptive network statistics is saved as a .png file. 

Therefore in it's most simple use, the tool is a'one-stop-shop' where the user only requires the profile link as an input and has comprehensible and publishable visualizations and tables as an output.

## Installation
This section includes detailed instructions for installing the tool, including any terminal commands that need to be executed and dependencies that need to be installed. Instructions should be understandable by non-technical users (e.g. someone who knows how to open a terminal and run commands, but isn't necessarily a programmer), for example:

1. Make sure you have Python version 3.8 or greater installed
2. Make sure you have R version 4.1.2 or greater installed

2. Download the tool's repository using the command:

        git clone https://github.com/digital-scrappy/network-analysis-hackathon

3. Move to the tool's directory and install the tool

        cd network-analysis-hackathon
        pip install .

## Usage
To use the tool, open the command line and enter

        network-analysis-hackathon https://twitter.com/YOURPROFILELINK

Expect the tool to take some time. The exact duration is dependent on the size of the profile, the computing power at your disposal and the speed of your internet connection. For large, highly active profiles, the scraping alone can take hours, add to that significant time for istalling required packages in R, the visualization and computation of the statistics. In general however, the computing power and patience should be the only limitation to the tool's ability to handle large data sets. 

You can spot errors in the process by looking at the output: If the network graph or the table is empty (or for the table shows 0 for values such as centrality) there clearly is an error in the process. You can verify if it is a matter of computing power by trying with a smaller profile. If the issue persists, it is most probably a fault in our code. Apologies. 

## Additional Information
In the future, we would like to increase the modularity of the tool by allowing users to not only use links to twitter profiles, but also uploading their own data or offering only the first part of the process: using a twitter link which scrapes data suitable for network analyses (i.e. creating an edgelist).

Furthermore, we would like to include error and warning messages as well as troubleshooting advice in the final output and/or during the process (e.g. if the process completes and produces 'meaningful' output, but this output is extremely unlikely given the network parameters). Such estimates of parameter likelihood could be computed using Exponential Random Graph Models (ERGMs) for statistical significance testing, or comparing the observed network to simulated Erdos-Renyi models. 

The plan is to move the network visualization and analysis completely to Python to achieve a more seamless integration, lower error rate and faster execution.

Currently, the biggest limitation is the limited troubleshooting ability in case the tool is (partially) dysfunctional. Offering a simple one-stop-shop tool to users with limited coding and command line experience, also comes at the cost of low customizability on the users' side. We did not have the time or expertise to build this tool as a GUI which offers customization. 
Experienced users however, can always modify the source code to their needs. 
