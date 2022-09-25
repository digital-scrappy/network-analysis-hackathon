from pathlib import Path
import pandas as pd
import plotly_express as px
import pandas
import networkx as nx
import numpy as np
import plotly.graph_objects as go

edge_df = pandas.read_csv(r'/home/timo/random_coding/Bellingcat Hackathon/edge_list.csv', header= None)
info_df = pandas.read_csv(r'/home/timo/random_coding/Bellingcat Hackathon/user_info.csv')

def plot_followers(user_info_df, output_dir : Path):
    follower_df = user_info_df.sort_values('#followers')
    follower_df = follower_df.tail(20)

    fig_follower = px.bar(follower_df, x='Unnamed: 0',y='#followers',
                        color= '#followers', color_continuous_scale= 'deep',
                labels={
                    "Unnamed: 0" : "username",
                    "#followers" : "number of followers"},
                        )
    fig_follower.update_traces(hovertemplate='username: %{x}'+'<br>followers: %{y}')
    fig_follower.update_layout(title_text= "accounts with most followers", title_x= 0.5)
    fig_follower.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig_follower.write_html(output_dir/ "follower_plot.html")

def plot_posts(user_info_df, output_dir : Path):
    post_df = user_info_df.sort_values('#posts')
    post_df = post_df.tail(20)

    fig_post = px.bar(post_df, x='Unnamed: 0', y='#posts',
                    color= '#posts', color_continuous_scale= 'deep',
                labels={
                    "Unnamed: 0" : "username",
                    "#posts" : "number of posts"})
    fig_post.update_traces(hovertemplate='username: %{x}'+'<br>posts: %{y}')
    fig_post.update_layout(title_text= "accounts with most posts", title_x= 0.5)
    fig_post.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    fig_post.write_html(output_dir / "post_plot.html")

def plot_analysis(g, user_info_df, output_dir : Path):
    n_users = user_info_df.shape[0]

    #degree_centrality = nx.degree_centrality(g)
    degree_centrality = dict(g.degree()) #this is raw and therefore what we want (I am not sure if is degree scientifically speaking)
    avg_centrality = round(sum(degree_centrality.values())/len(degree_centrality.values()),2)
    most_connections = round(max(degree_centrality.values()), 2)
    most_connected = max(degree_centrality, key=degree_centrality.get)

    try:
        path_length = nx.average_shortest_path_length(g) #get added here because this code will err if graph is not fully connected
    except networkx.exception.NetworkXError:
        print("graph is not fully connected")
        path_length = None
    ncomps = nx.number_connected_components(g)
    members_largest_c = max(nx.connected_components(g), key=len) #will see later what to do with this, not fitting for table
    size_larges_c = len(members_largest_c)
    prop_largest_comp = round((size_larges_c/n_users)*100, 2)

    analysis_data = np.array([str(n_users), str(avg_centrality),str(most_connected), str(most_connections), str(paths),
                            str(ncomps), str(size_larges_c), str(prop_largest_comp)])
    rows = ['number of users', 'average number of connections', 'most connected user', 'number of their connections',
            'average distance between two members', 'number of components', 'size of the largest component',
            'proportion of the largest component']
    fig_analysis = go.Figure(data=[go.Table( header= dict(fill_color= 'white'),
        cells=dict(values=[rows, analysis_data],
                line_color='white',
                fill_color='lightblue',
                align='left'))
    ])

    fig_analysis.write_html(output_dir / "descriptive_network_statistics.html")

    print('The network has ' + str(n_users) + ' members.' +
        ' \nThe average user has ' + str(avg_centrality) + " connections." +
        ' \nThe most connected user is ' + str(most_connected) + ' with ' + str(most_connections) + ' connections.' +
        ' \nThe network has ' + str(ncomps) + ' components, \nwhere the largest component has ' + str(size_larges_c) + ' members ' +
        ' and makes up ' + str(prop_largest_comp) + ' % of the network.' )

