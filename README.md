# Project name
## OUR PLAN

Departing from a python script that scrapes data on interactions between twitter users and general information on their profiles, we aim to create a GUI in which users can specify a twitter user name, a recursive search depth (as in how many 'steps away' the tool should go from the original interaction), and a number of tweets to download from the originally specified user. This data is then used to create interactive visualizations and tables in html format, containing descriptive statistics and other information about the network, which are automatically saved on the users machine. We aim to allow multiple queries at the same time, an option to save queries for future searches and the display of error messages if the input is misspecified or the files are faulty.
The tool should be modular in a way that it can easily be extended with additional features in the future, ranging from the GUI to the scripts for analysis and visualizations. The modularity would also allow researchers to not use the tool as a 'one-stop-shop', but also only use parts of the tool, such as either the scraping or the analysis seperately.

## TODO

- add Caveat that network will be plotted in a not fully connected way since we only count two way interactions
- update installation part

## Members

JH https://github.com/digital-scrappy

TD https://github.com/timo-damm

IP https://github.com/Ioana-P
## Dependencies
Python
Pandoc
gcc-fortan
python dependencies in requirements.txt

## Tool Description
In the [Bellingcat Survey](https://www.bellingcat.com/resources/2022/08/12/these-are-the-tools-open-source-researchers-say-they-need/) a majority researchers indicated they are in need of *free* tools that are *easy to use* (i.e. do not include writing your own code, using GitHub, or extensive use of the command line. With regards to network analysis specifically, users were in need of "*a network analysis tool to analyse social media connections*", which includes measures such as degree centrality and displays clear graphs. Other users were in need of "*a simple tool to visualize connections*" or "*a tool that helps draw network diagrams from data*"
The tool we developed produces easily understandable graphs and information about the network, without having to have knowledge on how to write code or use a command line interface. 
In it's most simple form, the user only needs to install and use the GUI. In the GUI, the user can then specify a project name for their project, give a twitter username, a recursive search depth and the number of tweets to be downloaded from the original user. The tool then automatically uses snscrape to scrape the profile data of the original user, such as  their user name, follower count, their friends count, their interactions, media uploads, retweets etc. Furthermore, the tweets are downloaded. An interaction between two profiles is identified when there is a mutual interaction (i.e. both users mention each other in at least one tweet). Then, the profile data and tweets of interacting profiles are scraped. This process is repeated n times, where n is the recursive search depth. 
Based on the retrieved interactions, the tool then assembles an edgelist which contains every interaction between this profile and other users, as well as the interactions two users that interacted with the original profile among each other. The edgelist is the primary data containing information about the network. The destination file for the edgelist is 'edgelist.csv' in the output directory of the user's machine. 
Other information, such as follower count, friends count, number of posts, verified status and profile description is assembled into a separate file.The destination file for the user information is 'user_info.csv' in the output directory of the user's machine.

Using the ```networkx``` python library, the tool then adds nodal characteristics from the separate dataset to the edgelist after transforming it into a network object. The resulting information-rich network is then used in the ```plotly``` python library, to create the visualizations and analyses. The output includes an interactive network graph (destination file: 'network_plot.html'), an interactive bar plot with the most followed accounts (destination file: 'follower_plot.html') an interactive bar plot with the acounts with the most posts (destination file: 'post_plot.html') and a table with descriptive network statistics, including the number of users, centrality measures, path length and information on components (destination file: 'descriptive_network_statistics.html') The descriptive statistics are partly adapted from Gerber and colleagues' (2013) reporting framework for network analyses (see [here](https://onlinelibrary.wiley.com/doi/pdf/10.1111/ajps.12011?casa_token=MTVxax7BWfkAAAAA:e6v3H2ciJlZT1BRuF1vauHmeuJnnGLjarp91CNuY2RaDMCC1x-awCF6iVQAtBLIr655VGFGXGyocXkBZ)).
All output files are saved in the output directory on the user's machine.  

Therefore in it's most simple use, the tool is a'one-stop-shop' where the user only requires the a username, a search depth and a number of tweets as an input and has comprehensible and publishable visualizations and tables as an output. A comprehensive documentation with screenshots of the GUI, error and success messages and examples of all outputs can be found [here](https://digital-scrappy.github.io/bellingcat_demopages/)

## Installation
This section includes detailed instructions for installing the tool, including any terminal commands that need to be executed and dependencies that need to be installed. Instructions should be understandable by non-technical users (e.g. someone who knows how to open a terminal and run commands, but isn't necessarily a programmer), for example:

1. Make sure you have Python version 3.8 or greater installed

2. Download the tool's repository using the command:

        git clone https://github.com/digital-scrappy/network-analysis-hackathon

3. Move to the tool's directory and install the tool

        cd network-analysis-hackathon
        pip install .

## Usage
To use the tool, open the GUI and specify the project name, user name, recursive search depth and number of tweets (part A). If multiple queries should be executed, upload a query table (query_table.csv) in part B to schedule the queries. The query table should include a column for each of the required inputs and a row for each query (i.e. SEARCH_USER, MAX_DEPTH, MAX_NUMBER_TWEETS). The query table should not contain a column with project names.

As the number of mentions in the network increases roughly exponentially with increases in the recursive search depth, it is worthwile thinking about the combination of search depth and number of tweets. Besides the longer runtime, the network plot also becomes increasingly messy. A combination that has worked well in the past is a recursive search depth of 2 with 50 tweets. This is obviously also dependent on how many users are tagged per tweet. 

The potential uses of the tool in practice are many. Firstly, it can be used to analyze and visualize networks around Twitter users to gain a deeper understanding of an online bubble, a political group or another network of actors. As many interactions take place in the digital sphere and getting a qualitative and quantitative overview of a certain group is practically impossible browsing Twitter, Blattodea is a powerful tool to get a glimpse of networks and make sense of interactions. The additional elements of the plots allow for practical uses such as identifying most central members (or hubs) of networks and demarcate them from peripherial members. Furthermore, it can give quantitative insights on the interaction between certain users. 
The usage of the tool does not end here though. As the retrieved data is cleaned and saved in a universal format it can be used across a variety of software solutions and for a variety of analyses. 

## Additional Information
In the future, we would like to increase the modularity of the tool by allowing users integrating and allowing users to choose from a higher variety of visualizations and analyses. We also plan on increasing modularity by integrating a customization of search parameters and the direct input of previously collected data in different formats to the tool.
An example of a new functionality to be introduced in the future is clustering based on topics. As the text of the tweets is downloaded, topic modeling and a clustering algorythm would enable visual clustering of the model by topics, giving more meta-information that would not easily be available when browsing Twitter. 
In the same vain, we plan on using exponential random graph models (ERGMs) in the future to statistically assess the most important topics or most important users in a network and if we see a clustering for example of people with similar political ideas.

Lastly, we plan on caching users for faster scraping.

Due to the high degree of integration, a limitation of the tool is the troubleshooting ability. There are several checks in place to prevent the tool from crashing (e.g. pre-analysis check of inputs in the GUI and corresponding error messages, optionality of code that is structure dependent), however if the tool becomes dysfunctional, editing the code is the only way of debugging, which is not feasible for all users.

If you have any questions, problems or additional input, please send us a message!
