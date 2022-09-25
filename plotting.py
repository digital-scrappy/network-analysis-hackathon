import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import plotly_express as px
import networkx as nx
from networkx.exception import NetworkXError
import numpy as np

def make_edge(x, y, text, width):
    
    '''Creates a scatter trace for the edge between x's and y's with given width

    Parameters
    ----------
    x    : a tuple of the endpoints' x-coordinates in the form, tuple([x0, x1, None])
    
    y    : a tuple of the endpoints' y-coordinates in the form, tuple([y0, y1, None])
    
    width: the width of the line

    Returns
    -------
    An edge trace that goes between x0 and x1 with specified width.
    '''
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'cornflowerblue'),
                       hoverinfo = 'text',
                       text      = ([text]),
                       mode      = 'lines')



def plot_network(graph, pos, edge_attrs, node_attrs, output_dir):
    G = graph

    #create edge trace
    edge_trace = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        x_pos = [x0, x1, None]
        y_pos = [y0, y1, None]
        
        num_interaction = edge_attrs[str((edge[0], edge[1]))]
        text = edge[0] + "-" + edge[1] 

        trace = make_edge(x_pos, y_pos, text, num_interaction * 100 + 1)
        edge_trace.append(trace)


    #create node trace
    node_trace = go.Scatter(x         = [],
                            y         = [],
                            text      = [],
                            hovertext = [],
                            textposition = "top center",
                            textfont_size = 20,
                            mode      = 'markers+text',
                            marker    = dict(color = [],
                                            size  = [],
                                            line  = None))



    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        color = "cornflowerblue" if node_attrs[node]["verified"] else "green"
        size  = node_attrs[node]["scaled_friends"] * 60 + 10
        verified = "verified" if node_attrs[node]["verified"] else "not verified"
        text = f"{node}\n {verified} \n #followers: {node_attrs[node]['#friends']}"
        node_trace["marker"]["color"] += (color,)
        node_trace["marker"]["size"] += (size,)
        node_trace['hovertext'] += (text,)
        node_trace['text'] += (node,)


    fig = go.Figure(layout=go.Layout(
                    title='Network plot',
                    # titlefont_size=20,
                    autosize=True,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    # annotations=[ dict(
                    #     text="annotation here",
                    #     showarrow=False,
                    #     xref="paper", yref="paper",
                    #     x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    for trace in edge_trace:
        fig.add_trace(trace)
    fig.add_trace(node_trace)
    fig.show()

    fig.write_html(output_dir / "network_plot.html")

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
        path_lenght = nx.average_shortest_path_length(g) #get added here because this code will err if graph is not fully connected
    except NetworkXError:
        print("graph is not fully connected")
        path_lenght = None
    ncomps = nx.number_connected_components(g)
    members_largest_c = max(nx.connected_components(g), key=len) #will see later what to do with this, not fitting for table
    size_larges_c = len(members_largest_c)
    prop_largest_comp = round((size_larges_c/n_users)*100, 2)

    analysis_data = np.array([str(n_users), str(avg_centrality),str(most_connected), str(most_connections), str(path_lenght),
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
