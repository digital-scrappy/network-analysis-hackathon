import plotly_express as px
import pandas
import networkx as nx

edge_df = pandas.read_csv('Data/2022-09-24_14-12/edge_list.csv', header= None)
user_info_df = pandas.read_csv('Data/2022-09-24_14-12/user_info.csv')

# def bar_plot_followers(info_df):

def plot_followers(user_info_df, output_dir):
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

#def bar_plot_posts(info_df):
def plot_posts(user_info_df, output_dir):
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

#def analysis_plot(info_df) :
def plot_analysis(g, user_info_df)
# g = nx.from_pandas_edgelist(df = edge_df, source = 0, target = 1) #unsure if we will need this in the final version
nusers = user_info_df.shape[0]
degree_centrality = nx.degree_centrality(g)


most_connected = [(key, value) for key, value in degree_centrality.items() if value == max(degree_centrality.values())]
print(most_connected)





#centrality? most acive user? density? number of  (meaningful) clusters?
